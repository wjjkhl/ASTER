import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from config import Config


class EnhancedCognitionBasedKnowledgeDistillation(nn.Module):
    """Enhanced knowledge distillation with dual matching for layer skipping"""

    def __init__(self, tau_kd=1.0, lambda_cog=Config.COG_WEIGHT, lambda_rep=Config.REP_WEIGHT):
        """
        Initialize knowledge distillation loss

        Args:
            tau_kd: Temperature for token importance calculation
            lambda_cog: Weight for cognitive consistency loss
            lambda_rep: Weight for representation loss
        """
        super(EnhancedCognitionBasedKnowledgeDistillation, self).__init__()
        self.tau_kd = tau_kd
        self.lambda_cog = lambda_cog
        self.lambda_rep = lambda_rep

    def compute_token_importance(self, h_cog, h_t):
        """
        Compute token importance based on similarity to <hcog> token

        Args:
            h_cog: <hcog> token representation [batch_size, hidden_size]
            h_t: All token representations [batch_size, seq_len, hidden_size]
        """
        similarity = torch.matmul(h_cog.unsqueeze(1), h_t.transpose(1, 2))
        similarity = similarity.squeeze(1) / self.tau_kd
        return torch.sigmoid(similarity)

    def forward(self, student_outputs, teacher_outputs, layer_path):
        """
        Compute enhanced knowledge distillation loss

        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            layer_path: List of layers selected by student model
        """

        student_device = next(iter(student_outputs['hidden_states'].values())).device


        input_cog_loss = 0.0  # Input matching cognitive loss
        input_rep_loss = 0.0  # Input matching representation loss
        output_cog_loss = 0.0  # Output matching cognitive loss
        output_rep_loss = 0.0  # Output matching representation loss

        # Process each decision point in the student's path
        for i in range(len(layer_path) - 1):
            s_curr_layer = layer_path[i]
            s_next_layer = layer_path[i + 1]

            # Skip embedding layer
            if s_curr_layer == 0 and s_next_layer == 1:
                continue

            # 1. INPUT MATCHING: Student model layer j input should match teacher model layer j-1 output
            t_input_target_layer = s_next_layer - 1

            if t_input_target_layer >= 0 and t_input_target_layer in teacher_outputs['hidden_states']:
                if 'layer_inputs' in student_outputs and s_next_layer in student_outputs['layer_inputs']:
                    s_input = student_outputs['layer_inputs'][s_next_layer]  # Student layer j input
                    s_input_cls = student_outputs['input_cls_tokens'][
                        s_next_layer]  # Student layer j input <hcog> token

                    t_input_hidden = teacher_outputs['hidden_states'][t_input_target_layer].to(
                        student_device)  # Teacher layer j-1 output
                    t_input_cls = teacher_outputs['cls_tokens'][
                        t_input_target_layer].to(student_device)  # Teacher layer j-1 output <hcog> token

                    # Cognitive consistency loss for input matching
                    in_cognitive_loss = F.mse_loss(s_input_cls, t_input_cls)
                    input_cog_loss += in_cognitive_loss

                    # Token importance for representation loss
                    t_in_importance = self.compute_token_importance(t_input_cls, t_input_hidden)
                    s_in_importance = self.compute_token_importance(s_input_cls, s_input)

                    # Importance difference
                    in_importance_diff = torch.abs(t_in_importance - s_in_importance)

                    # Token-level loss
                    in_token_loss = F.mse_loss(s_input, t_input_hidden, reduction='none')
                    in_token_loss = in_token_loss.mean(dim=-1)  # Average over feature dimension
                    in_weighted_loss = (in_importance_diff * in_token_loss).mean()

                    input_rep_loss += in_weighted_loss

            # 2. OUTPUT MATCHING: Student model layer j output should match teacher model layer j output
            t_output_target_layer = s_next_layer

            if t_output_target_layer in teacher_outputs['hidden_states']:
                s_output = student_outputs['hidden_states'][s_next_layer]  # Student layer j output

                t_output = teacher_outputs['hidden_states'][t_output_target_layer].to(
                    student_device)  # Teacher layer j output

                s_output_cls = student_outputs['cls_tokens'][s_next_layer]  # Student layer j output <hcog> token
                t_output_cls = teacher_outputs['cls_tokens'][
                    t_output_target_layer].to(student_device)  # Teacher layer j output <hcog> token

                # Cognitive consistency loss for output matching
                out_cognitive_loss = F.mse_loss(s_output_cls, t_output_cls)
                output_cog_loss += out_cognitive_loss

                # Token importance for representation loss
                t_out_importance = self.compute_token_importance(t_output_cls, t_output)
                s_out_importance = self.compute_token_importance(s_output_cls, s_output)

                # Importance difference
                out_importance_diff = torch.abs(t_out_importance - s_out_importance)

                # Token-level loss
                out_token_loss = F.mse_loss(s_output, t_output, reduction='none')
                out_token_loss = out_token_loss.mean(dim=-1)  # Average over feature dimension
                out_weighted_loss = (out_importance_diff * out_token_loss).mean()

                output_rep_loss += out_weighted_loss

        # Total distillation loss
        total_loss = self.lambda_cog * (input_cog_loss + output_cog_loss) + \
                     self.lambda_rep * (input_rep_loss + output_rep_loss)

        return total_loss, input_cog_loss + output_cog_loss, input_rep_loss + output_rep_loss


class TemporalImportanceDifferenceReward:
    """Implementation of Temporal Importance Difference Reward (TIDR)"""

    def __init__(self, total_layers, beta=Config.TIDR_BETA):
        """
        Initialize TIDR calculator

        Args:
            total_layers: Total number of layers in the model (L)
            beta: Balancing parameter for computational efficiency reward
        """
        self.total_layers = total_layers
        self.beta = beta

    def compute_temporal_weight(self, t, T, l_next, l_curr):
        """
        Compute temporal importance weight

        Args:
            t: Current decision step
            T: Total number of decision steps
            l_next: Next layer index
            l_curr: Current layer index
        """

        time_factor = float(T - t) / float(T)

        # 将层索引转换为浮点数张量
        l_next_tensor = torch.tensor(float(l_next), dtype=torch.float32)
        l_curr_tensor = torch.tensor(float(l_curr), dtype=torch.float32)
        total_layers_tensor = torch.tensor(float(self.total_layers), dtype=torch.float32)

        skip_factor = torch.sigmoid((l_next_tensor - l_curr_tensor) / total_layers_tensor)

        # 返回时间权重
        return time_factor * skip_factor.item()

    def compute_efficiency_reward(self, l_next, l_curr):
        """
        Compute computational efficiency reward

        Args:
            l_next: Next layer index
            l_curr: Current layer index
        """
        # Reward is proportional to number of skipped layers, but only if not exiting
        if l_next != self.total_layers:
            return self.beta * (l_next - l_curr - 1)
        return 0.0

    def compute_reward(self, final_reward, t, T, l_next, l_curr):
        """
        Compute the complete TIDR reward

        Args:
            final_reward: Reward based on final prediction (±1)
            t: Current decision step
            T: Total number of decision steps
            l_next: Next layer index
            l_curr: Current layer index
        """
        # 计算时间权重
        w_t = self.compute_temporal_weight(t, T, l_next, l_curr)

        # 计算效率奖励
        r_comp = self.compute_efficiency_reward(l_next, l_curr)

        if isinstance(final_reward, torch.Tensor):
            final_reward = final_reward.item()

        # 组合奖励
        reward = final_reward * w_t + r_comp

        return reward


class TotalLoss(nn.Module):
    """Combined generation, knowledge distillation, and policy gradient loss"""

    def __init__(self, num_layers, kd_weight=Config.KD_WEIGHT, policy_weight=Config.POLICY_WEIGHT):
        """
        Initialize combined loss function

        Args:
            num_layers: Number of model layers
            kd_weight: Weight for knowledge distillation loss
            policy_weight: Weight for policy gradient loss
        """
        super(TotalLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = EnhancedCognitionBasedKnowledgeDistillation()
        self.kd_weight = kd_weight
        self.policy_weight = policy_weight
        self.tidr = TemporalImportanceDifferenceReward(num_layers)

    def forward(self, student_outputs, teacher_outputs, targets, layer_path, t, T):
        """
        Compute total loss

        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            targets: Ground truth token IDs (yes/no的token IDs)
            layer_path: List of layers selected by student model
            t: Current decision step
            T: Total number of decision steps
        """

        student_device = next(iter(student_outputs['hidden_states'].values())).device

        targets = targets.to(student_device)

        # 处理类别不平衡：在CE中为正类(yes=1)设置权重≈0.61，负类(no=0)=1.0
        ce_class_weight = torch.tensor([1.0, 0.61], device=student_device, dtype=student_outputs['logits'].dtype)
        ce_unscaled = F.cross_entropy(student_outputs['logits'], targets, weight=ce_class_weight)
        pred_loss = Config.TASK_CE_WEIGHT * ce_unscaled

        # Knowledge distillation loss
        kd_loss, cog_loss, rep_loss = self.kd_loss(student_outputs, teacher_outputs, layer_path)

        # Calculate prediction accuracy
        student_preds = torch.argmax(student_outputs['logits'], dim=1)
        accuracy = (student_preds == targets).float().mean()


        path_length = len(layer_path)
        path_length_penalty = 0.0
        if path_length < Config.MIN_PATH_LENGTH:
            path_length_penalty = 0.05 * (Config.MIN_PATH_LENGTH - path_length) / Config.MIN_PATH_LENGTH
        path_length_penalty = torch.tensor(path_length_penalty, device=student_device)

        # Calculate scalar reward (in policy gradient, reward doesn't need gradients)
        final_reward = 2 * accuracy.detach().item() - 1  # Map to [-1,1] range

        # Calculate rewards for each decision point
        rewards = []
        for i in range(len(layer_path) - 1):
            l_curr = layer_path[i]
            l_next = layer_path[i + 1]
            reward = self.tidr.compute_reward(final_reward, t, T, l_next, l_curr)
            rewards.append(reward)

        # Policy gradient loss
        policy_loss = torch.tensor(0.0, device=student_device)

        if 'log_probs' in student_outputs and student_outputs['log_probs']:
            log_probs = student_outputs['log_probs']

            # 对齐长度：首跳(0->1)没有log_prob，rewards比log_probs多1时，丢弃首个reward
            if len(rewards) == len(log_probs) + 1:
                rewards = rewards[1:]
            # 其他异常情况：截断到最短长度
            if len(log_probs) != len(rewards):
                min_len = min(len(log_probs), len(rewards))
                log_probs = log_probs[:min_len]
                rewards = rewards[:min_len]

            if len(log_probs) > 0:
                pg_losses = []
                for log_prob, reward in zip(log_probs, rewards):
                    reward_tensor = torch.tensor(reward, device=student_device, dtype=torch.float32).detach()
                    if log_prob.requires_grad:
                        pg_loss = -log_prob * reward_tensor
                    else:
                        log_prob = log_prob.clone().detach().requires_grad_(True)
                        pg_loss = -log_prob * reward_tensor
                    pg_losses.append(pg_loss)
                policy_loss = torch.stack(pg_losses).sum()

        # Total loss with path length penalty
        total_loss = pred_loss + self.kd_weight * kd_loss + self.policy_weight * policy_loss + path_length_penalty


        if torch.isnan(total_loss):
            print("警告: 检测到NaN损失，使用备用损失计算")
            total_loss = pred_loss  # 使用只有预测损失的备用计算

        return {
            'total_loss': total_loss,
            'pred_loss': pred_loss,
            'kd_loss': kd_loss,
            'policy_loss': policy_loss,
            'cog_loss': cog_loss,
            'rep_loss': rep_loss,
            'path_length_penalty': path_length_penalty,
            'rewards': rewards,
            'accuracy': accuracy
        }
