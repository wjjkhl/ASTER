import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
from config import Config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, BackwardPrefetch, ShardingStrategy
)
# 修改models.py中的导入部分
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, BackwardPrefetch, ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy  # 使用transformer_auto_wrap_policy替代default_auto_wrap_policy
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
try:
    # For broad transformers compatibility
    from transformers.models.llama.modeling_llama import make_causal_mask as llama_make_causal_mask
    from transformers.models.llama.modeling_llama import expand_attention_mask as llama_expand_attention_mask
except Exception:
    llama_make_causal_mask = None
    llama_expand_attention_mask = None


class DynamicSkipDistanceAdapter(nn.Module):
    """Dynamic skip distance adapter that adjusts complexity based on skip distance"""

    def __init__(self, hidden_size, max_distance=Config.MAX_DISTANCE,
                 adapter_base_size=Config.ADAPTER_BASE_SIZE, max_blocks=Config.MAX_BLOCKS):
        super(DynamicSkipDistanceAdapter, self).__init__()
        self.hidden_size = hidden_size
        self.max_distance = max_distance
        self.max_blocks = max_blocks

        # Distance embedding
        self.distance_embedding = nn.Embedding(max_distance + 1, adapter_base_size)

        # Down and up projections
        self.down_proj = nn.Linear(hidden_size, adapter_base_size * 4)
        self.up_proj = nn.Linear(adapter_base_size * 4, hidden_size)

        # Basic transform block
        self.transform_block = nn.Sequential(
            nn.Linear(adapter_base_size * 4, adapter_base_size * 4),
            nn.LayerNorm(adapter_base_size * 4),
            nn.GELU()
        )

        # Gating mechanism
        self.gate = nn.Linear(hidden_size + adapter_base_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize with small weights for stability
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize gate bias toward preserving original information
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, hidden_states, source_layer, target_layer):
        """
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            source_layer: Source layer index
            target_layer: Target layer index
        """
        batch_size, seq_len, _ = hidden_states.shape
        # 保证输入与权重dtype一致，避免bf16/float32不匹配
        if hasattr(self.down_proj, 'weight'):
            target_dtype = self.down_proj.weight.dtype
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)
        residual = hidden_states

        # Calculate skip distance
        distance = min(target_layer - source_layer, self.max_distance)

        # Get distance embedding
        dist_embed = self.distance_embedding(torch.tensor(distance, device=hidden_states.device))
        dist_embed = dist_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Down projection
        x = self.down_proj(hidden_states)

        # Skip distance adaptive processing - apply n blocks
        n_blocks = min(self.max_blocks, 1 + int(distance * 0.5))
        for _ in range(n_blocks):
            x = self.transform_block(x)

        # Up projection
        x = self.up_proj(x)

        # Calculate gate values
        gate_input = torch.cat([residual, dist_embed], dim=-1)
        gate_value = torch.sigmoid(self.gate(gate_input))

        # Gated mixing
        output = gate_value * x + (1 - gate_value) * residual

        # Layer normalization
        output = self.layer_norm(output)

        return output


class LayerEmbedding(nn.Module):
    """Embedding function for layer positions"""

    def __init__(self, max_layers, embedding_dim):
        super(LayerEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_layers + 1, embedding_dim)

        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, layer_index):
        # 针对 Python int 直接在GPU上切片，避免反复构造设备张量造成阻塞
        if isinstance(layer_index, int):
            embedding = self.embedding.weight[layer_index]
        else:
            # 确保索引为long并在与权重相同的设备上
            if not isinstance(layer_index, torch.Tensor):
                layer_index = torch.tensor(layer_index, dtype=torch.long, device=self.embedding.weight.device)
            else:
                if layer_index.dtype != torch.long:
                    layer_index = layer_index.long()
                if layer_index.device != self.embedding.weight.device:
                    layer_index = layer_index.to(self.embedding.weight.device, non_blocking=True)

            embedding = self.embedding(layer_index)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        return embedding


class ScoringModel(nn.Module):
    """MLP-based scoring model for layer skipping decisions"""

    def __init__(self, hidden_size, layer_embed_dim, mlp_hidden_dim=256, mlp_intermediate_dim=128):
        super(ScoringModel, self).__init__()
        self.input_dim = hidden_size + 2 * layer_embed_dim

        # MLP layers
        self.W1 = nn.Linear(self.input_dim, mlp_hidden_dim)
        self.W2 = nn.Linear(mlp_hidden_dim, mlp_intermediate_dim)
        self.W3 = nn.Linear(mlp_intermediate_dim, 1)
        self.activation = nn.GELU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W1.weight, nonlinearity='relu')
        nn.init.zeros_(self.W1.bias)

        nn.init.kaiming_normal_(self.W2.weight, nonlinearity='relu')
        nn.init.zeros_(self.W2.bias)

        nn.init.normal_(self.W3.weight, std=0.02)
        nn.init.zeros_(self.W3.bias)

    def forward(self, h_cog, l_curr_embed, l_cand_embed, l_curr=None, l_cand=None, path_length=None):
        batch_size = h_cog.size(0)

        # Ensure embeddings match batch size
        if l_curr_embed.size(0) == 1:
            l_curr_embed = l_curr_embed.expand(batch_size, -1)
        if l_cand_embed.size(0) == 1:
            l_cand_embed = l_cand_embed.expand(batch_size, -1)

        # Concatenate inputs
        x = torch.cat([h_cog, l_curr_embed, l_cand_embed], dim=-1)

        # Apply MLP layers
        x = self.activation(self.W1(x))
        x = self.activation(self.W2(x))
        score = self.W3(x)

        # 应用路径长度惩罚 - 如果提供了必要的参数
        if path_length is not None and l_curr is not None and l_cand is not None:
            if path_length < Config.MIN_PATH_LENGTH:
                # 计算跳跃大小
                jump_size = l_cand - l_curr
                # 路径长度惩罚
                length_penalty = Config.LENGTH_PENALTY_ALPHA * jump_size * (
                            Config.MIN_PATH_LENGTH - path_length) / Config.MIN_PATH_LENGTH
                # 转换为张量并确保维度正确
                if isinstance(length_penalty, (int, float)):
                    length_penalty = torch.tensor(length_penalty, device=score.device, dtype=score.dtype)
                length_penalty = length_penalty.expand_as(score)
                # 应用惩罚
                score = score - length_penalty

        return score.squeeze(-1)


class StudentLLAMA(nn.Module):
    """Student LLAMA model with dynamic layer skipping capabilities"""

    def __init__(self, base_model_path, layer_embed_dim=Config.LAYER_EMBED_DIM,
                 adapter_base_size=Config.ADAPTER_BASE_SIZE, device=torch.device("cuda:0"),
                 use_fsdp=False):
        super(StudentLLAMA, self).__init__()

        self.use_fsdp = use_fsdp
        self.device = device

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            padding_side="right"  # 与baseline对齐：右侧padding
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"已将学生模型分词器的eos_token设置为pad_token: {self.tokenizer.pad_token}")

        # 不再依赖<hcog>特殊token；认知锚点将由最后非pad位置替代

        # Load base model with full precision
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  # 改为bf16权重
            device_map={'': device},
            use_cache=False,
            output_hidden_states=True
        )

        # 如果未添加特殊token，则无需resize token embeddings

        # Extract config
        self.config = self.base_model.config
        self.num_layers = len(self.base_model.model.layers)
        self.hidden_size = self.config.hidden_size

        # 与baseline对齐：使用带前导空格的" yes"与" no"作为分类标签token
        self.yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        self.token_to_bool = {
            self.yes_token_id: True,
            self.no_token_id: False
        }
        print(f"Token映射: yes={self.yes_token_id}, no={self.no_token_id}")

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Initialize dynamic skip adapter
        self.dynamic_adapter = DynamicSkipDistanceAdapter(
            hidden_size=self.hidden_size,
            max_distance=Config.MAX_DISTANCE,
            adapter_base_size=adapter_base_size,
            max_blocks=Config.MAX_BLOCKS
        )

        # Layer embedding
        self.layer_embedding = LayerEmbedding(self.num_layers, layer_embed_dim)

        # Scoring model for layer skipping decisions
        self.scoring_model = ScoringModel(
            hidden_size=self.hidden_size,
            layer_embed_dim=layer_embed_dim
        )

        # Move trainable components to device并统一到bf16以匹配base_model输出
        self.dynamic_adapter.to(device=device, dtype=torch.bfloat16)
        self.layer_embedding.to(device=device, dtype=torch.bfloat16)
        self.scoring_model.to(device=device, dtype=torch.bfloat16)

        # Skip statistics
        self.skip_stats = {}

        # Policy gradient: save log probabilities of actions
        self.saved_log_probs = []

        # 如果启用FSDP，使用FSDP包装模型
        if self.use_fsdp and dist.is_initialized():
            # 定义包装策略，针对Transformer层
            my_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    LlamaDecoderLayer,  # 包装LLAMA的解码器层
                }
            )

            # 仅包装基础模型
            # 忽略不参与分片的模块：embedding、final norm、lm_head（便于手动逐层执行时直接访问）
            ignored = []
            try:
                ignored.append(self.base_model.model.embed_tokens)
            except Exception:
                pass
            try:
                ignored.append(self.base_model.model.norm)
            except Exception:
                pass
            try:
                ignored.append(self.base_model.lm_head)
            except Exception:
                pass

            self.base_model = FSDP(
                self.base_model,
                auto_wrap_policy=my_auto_wrap_policy,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                cpu_offload=CPUOffload(offload_params=False),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=device.index,
                limit_all_gathers=True,
                ignored_modules=tuple(ignored) if ignored else None
            )

            # 解冻适配器层参数 - 确保只训练这些参数
            for param in self.base_model.parameters():
                param.requires_grad = False

            # 不再对可训练模块进行FSDP包装，以便直接保存/加载其state_dict

        # Print trainable parameter count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Student model has {trainable_params:,} trainable parameters")
        print(f"学生模型是否使用FSDP: {self.use_fsdp}")

    def select_next_layer(self, h_cog, l_curr, layer_path, temperature=1.0, training=True, max_skip_distance=None):
        """Select the next layer based on scoring model with path length constraints"""
        # 获取当前层嵌入
        l_curr_embed = self.layer_embedding(l_curr)

        # 获取批次大小
        batch_size = h_cog.size(0)

        # 特殊情况：嵌入层(0)总是继续到第一个transformer层(1)
        if l_curr == 0:
            return 1, None

        # 计算当前路径长度
        path_length = len(layer_path)

        # 计算剩余需要的层数
        remaining_needed = max(0, Config.MIN_PATH_LENGTH - path_length)

        # 计算剩余可用层数
        remaining_available = self.num_layers - l_curr

        # 计算最大允许跳过的层数
        max_allowed_skip = max(1, remaining_available - remaining_needed)

        # 计算有效最大跳跃距离
        effective_max_distance = min(max_skip_distance or float('inf'), max_allowed_skip)

        scores_list = []
        candidates = []

        # 计算所有候选层的评分
        start_idx = l_curr + 1
        end_idx = min(self.num_layers + 1, l_curr + effective_max_distance + 1)

        for l_cand in range(start_idx, end_idx):
            l_cand_embed = self.layer_embedding(l_cand)

            # 评分时加入路径长度惩罚
            score = self.scoring_model(
                h_cog,
                l_curr_embed,
                l_cand_embed,
                l_curr=l_curr,
                l_cand=l_cand,
                path_length=path_length
            )

            scores_list.append(score)
            candidates.append(l_cand)

        # 堆叠评分
        if not scores_list:  # 没有候选层(在实践中不应该发生)
            return l_curr + 1, None

        scores = torch.stack(scores_list, dim=1)

        # 防止数值不稳定
        if torch.isnan(scores).any():
            # 如果检测到NaN，使用默认行为 - 选择下一层
            print("警告: 检测到NaN评分，回退到线性层执行")
            return l_curr + 1, None

        # 选择下一层时
        if training:
            # 使用策略梯度方法
            mean_scores = scores.mean(dim=0)

            # 防止数值不稳定性
            mean_scores = torch.nan_to_num(mean_scores, nan=-1e10)

            # 以logits形式构建分布（转float32避免bf16概率Simplex误差）
            logits = (mean_scores / temperature).to(torch.float32)
            m = torch.distributions.Categorical(logits=logits)

            # 采样动作
            action = m.sample()

            # 保存对数概率用于策略梯度
            log_prob = m.log_prob(action)
            log_prob = log_prob.clone()  # 确保创建新的张量
            self.saved_log_probs.append(log_prob)

            # 获取选择的层索引
            idx = action.item()
            return candidates[idx], log_prob
        else:
            # 推理期间贪婪选择
            mean_scores = scores.mean(dim=0)

            # 防止数值不稳定性
            mean_scores = torch.nan_to_num(mean_scores, nan=-1e10)

            idx = torch.argmax(mean_scores).item()
            return candidates[idx], scores[:, idx]

    def forward(self, input_ids, attention_mask=None, temperature=1.0, training=True,
                true_skipping=False, max_skip_distance=None, enforce_min_path=False):
        """Forward pass that truly executes only the selected layers (real skipping)."""
        batch_size, seq_length = input_ids.size()

        # Prepare attention mask and embeddings
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        # LLaMA components
        llama_model = self.base_model.model
        hidden_states = llama_model.embed_tokens(input_ids)

        # Prepare position ids and causal mask consistent with LLaMA
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.to(dtype=torch.long, device=hidden_states.device)
        # Prepare 4D attention mask compatible with LLaMA layers
        if llama_make_causal_mask is not None and llama_expand_attention_mask is not None:
            dtype = hidden_states.dtype
            device = hidden_states.device
            causal_mask = llama_make_causal_mask((batch_size, seq_length), dtype=dtype, device=device, past_key_values_length=0)
            if attention_mask is not None:
                expanded_attn_mask = llama_expand_attention_mask(attention_mask, dtype, tgt_len=seq_length)
                causal_mask = causal_mask + expanded_attn_mask
        else:
            # Fallback: simple causal mask without padding expansion
            dtype = hidden_states.dtype
            device = hidden_states.device
            mask = torch.full((seq_length, seq_length), float("-inf"), device=device, dtype=dtype)
            mask = torch.triu(mask, diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)

        # 使用“最后非pad”位置作为认知锚点（替代<hcog>）
        anchor_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=self.device)

        # 初始化跟踪变量
        layer_path = [0]  # Start with embedding layer index 0
        detailed_path = []
        self.skip_stats = {"skipped_layers": [], "visited_layers": [0]}
        log_probs = []

        # 清除之前的log probabilities
        if training:
            self.saved_log_probs = []

        # 取锚点表征（每个样本的最后非pad位置）
        h_cog = hidden_states[batch_indices, anchor_positions]

        processed_hidden_states = {0: hidden_states}
        all_cls_tokens = {0: h_cog}
        all_layer_inputs = {0: hidden_states}
        all_input_cls_tokens = {0: h_cog}

        # 先强制处理第一层（层索引1，对应模块索引0）
        l_curr = 0
        l_next = 1
        # 记录输入给第1层
        all_layer_inputs[l_next] = hidden_states
        all_input_cls_tokens[l_next] = h_cog
        # 执行第1层
        layer_module = llama_model.layers[l_next - 1]
        layer_out = layer_module(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        # 状态记录
        layer_path.append(l_next)
        self.skip_stats["visited_layers"].append(l_next)
        processed_hidden_states[l_next] = hidden_states
        h_cog = hidden_states[batch_indices, anchor_positions]
        all_cls_tokens[l_next] = h_cog
        detailed_path.append("Processing layer 1 (required)")
        l_curr = l_next

        # 最小路径长度（可选）
        min_required_path = Config.MIN_PATH_LENGTH if enforce_min_path else 0

        # 继续处理其余层
        while l_curr < self.num_layers:
            current_path_length = len(layer_path)
            remaining_layers = self.num_layers - l_curr

            if current_path_length < min_required_path and remaining_layers > 0:
                min_remaining_needed = max(0, min_required_path - current_path_length)
                effective_max_skip = min(
                    max_skip_distance or float('inf'),
                    max(1, remaining_layers - min_remaining_needed)
                )
            else:
                effective_max_skip = max_skip_distance

            # 选择下一层
            l_next, log_prob = self.select_next_layer(
                h_cog, l_curr, layer_path, temperature, training, effective_max_skip
            )
            if log_prob is not None:
                log_probs.append(log_prob)

            if true_skipping and l_next > l_curr + 1:
                # 真正的层跳跃 - 用适配器桥接，并仅执行目标层
                adapted_hidden = self.dynamic_adapter(processed_hidden_states[l_curr], l_curr, l_next)

                # 记录跳过层
                skipped = list(range(l_curr + 1, l_next))
                self.skip_stats["skipped_layers"].extend(skipped)
                detailed_path.append(f"Skipping from {l_curr} to {l_next}, skipped {skipped}")

                # 记录输入给目标层
                all_layer_inputs[l_next] = adapted_hidden
                all_input_cls_tokens[l_next] = adapted_hidden[batch_indices, anchor_positions]

                # 执行目标层（模块索引 l_next-1）
                layer_module = llama_model.layers[l_next - 1]
                layer_out = layer_module(
                    adapted_hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

                # 记录输出与路径
                layer_path.append(l_next)
                self.skip_stats["visited_layers"].append(l_next)
                processed_hidden_states[l_next] = hidden_states

                # 更新 h_cog
                h_cog = hidden_states[batch_indices, anchor_positions]
                all_cls_tokens[l_next] = h_cog
            else:
                # 非跳跃：逐层执行到目标层（当 true_skipping=False 或 l_next=l_curr+1）
                detailed_path.append(f"Regular processing from {l_curr} to {l_next}")
                for layer_idx in range(l_curr + 1, l_next + 1):
                    if layer_idx <= self.num_layers:
                        # 记录输入给该层
                        all_layer_inputs[layer_idx] = hidden_states
                        all_input_cls_tokens[layer_idx] = hidden_states[batch_indices, anchor_positions]

                        # 执行该层（模块索引 layer_idx-1）
                        layer_module = llama_model.layers[layer_idx - 1]
                        layer_out = layer_module(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False
                        )
                        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

                        # 记录输出与路径
                        layer_path.append(layer_idx)
                        self.skip_stats["visited_layers"].append(layer_idx)
                        processed_hidden_states[layer_idx] = hidden_states

                        # 更新 h_cog
                        h_cog = hidden_states[batch_indices, anchor_positions]
                        all_cls_tokens[layer_idx] = h_cog

            # 更新当前层
            l_curr = l_next
            if l_next == self.num_layers:
                break

        final_hidden_state = processed_hidden_states[layer_path[-1]]

        # 生成下一个token的logits：允许梯度通过 final_hidden_state 回流到适配器/决策路径
        llama_model = self.base_model.model
        if hasattr(llama_model, 'norm') and llama_model.norm is not None:
            normed = llama_model.norm(final_hidden_state)
        else:
            normed = final_hidden_state
        lm_logits = self.base_model.lm_head(normed)

        # 仅保留 yes/no 的二分类 logits
        # 使用最后非pad位置抽取 yes/no logits，确保与baseline一致
        batch_indices = torch.arange(batch_size, device=self.device)
        no_vals = lm_logits[batch_indices, anchor_positions, self.no_token_id]
        yes_vals = lm_logits[batch_indices, anchor_positions, self.yes_token_id]
        yes_no_logits = torch.stack((no_vals, yes_vals), dim=1)

        return {
            'logits': yes_no_logits,
            'lm_logits': lm_logits[:, -1, :],
            'layer_path': layer_path,
            'log_probs': log_probs,
            'hidden_states': processed_hidden_states,
            'cls_tokens': all_cls_tokens,
            'layer_inputs': all_layer_inputs,
            'input_cls_tokens': all_input_cls_tokens,
            'detailed_path': detailed_path
        }

    def generate(self, input_ids, attention_mask=None, temperature=0.1, max_new_tokens=1):
        """生成模式，限制最大生成token数为1"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                training=False,
                true_skipping=True
            )

            # 从二分类logits中获取预测
            yes_no_logits = outputs['logits']
            next_token_probs = F.softmax(yes_no_logits, dim=-1)
            class_indices = torch.argmax(next_token_probs, dim=-1)

            # 将类别索引(0,1)映射回真实token ID(2201,9891)
            token_ids = torch.tensor(
                [self.no_token_id if idx == 0 else self.yes_token_id for idx in class_indices],
                device=self.device
            )

            # 组合原始输入和生成的token
            combined_ids = torch.cat([input_ids, token_ids.unsqueeze(1)], dim=1)

            return combined_ids, next_token_probs

    def get_skip_stats(self):
        """Return layer skipping statistics from the last forward pass"""
        return self.skip_stats


class TeacherLLAMA(nn.Module):
    """Teacher LLAMA model that processes all layers"""

    def __init__(self, base_model_path, device=torch.device("cuda:0")):
        super(TeacherLLAMA, self).__init__()

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            padding_side="right"  # 与baseline对齐：右侧padding
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"已将教师模型分词器的eos_token设置为pad_token: {self.tokenizer.pad_token}")

        # 不再依赖<hcog>特殊token；使用最后非pad位置作为锚点
        num_added = 0

        # Load base model with full precision
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,  # 改为bf16权重
            device_map={'': device},
            use_cache=False,
            output_hidden_states=True
        )

        # 未添加特殊token时无需resize token embeddings

        # Get configuration
        self.config = self.base_model.config
        self.num_layers = len(self.base_model.model.layers)

        # 与baseline对齐（带前导空格）
        self.yes_token_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        print(f"Teacher Token映射: yes={self.yes_token_id}, no={self.no_token_id}")

        # Freeze all parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Set device
        self.device = device
        print(f"Teacher model loaded on {device}")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through all layers, returning hidden states for distillation"""
        # 直接使用基础模型的前向传播获取所有隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 获取所有层的隐藏状态
        hidden_states = outputs.hidden_states

        # 使用最后非pad位置作为锚点
        anchor_positions = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=self.device)

        # 提取每层的<hcog>表示
        cls_tokens = {}
        for i, layer_output in enumerate(hidden_states):
            cls_tokens[i] = layer_output[batch_indices, anchor_positions]

        # 获取最终输出
        last_hidden_state = hidden_states[-1]

        # 与Student对齐：在lm_head前应用最终RMSNorm（若存在）
        llama_model = self.base_model.model
        if hasattr(llama_model, 'norm') and llama_model.norm is not None:
            normed = llama_model.norm(last_hidden_state)
        else:
            normed = last_hidden_state

        # 使用LM头生成next token的logits
        lm_logits = self.base_model.lm_head(normed)

        # 只关注"yes"和"no" token的logits
        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=self.device)
        no_vals = lm_logits[batch_indices, anchor_positions, self.no_token_id]
        yes_vals = lm_logits[batch_indices, anchor_positions, self.yes_token_id]
        yes_no_logits = torch.stack((no_vals, yes_vals), dim=1)

        # 将hidden_states转换为字典形式
        hidden_states_dict = {i: hidden_states[i] for i in range(len(hidden_states))}

        return {
            'logits': yes_no_logits,  # yes/no logits形状为[batch_size, 2]
            'lm_logits': lm_logits[:, -1, :],  # 完整的语言模型logits，仅最后一个token位置
            'hidden_states': hidden_states_dict,
            'cls_tokens': cls_tokens
        }