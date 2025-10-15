# /aster-bert/components.py
# NO CHANGES NEEDED. This file is model-agnostic.

import torch
import torch.nn as nn
# Import from the new config file for consistency if running standalone tests,
# but for the project, the DEVICE will be passed or sourced from the main config.
try:
    from config_bert import DEVICE
except ImportError:
    from config import DEVICE


class DynamicAdapter(nn.Module):
    """
    A lightweight dynamic adapter that conditions its transformation on the
    current and next layer indices. It is initialized with the same dtype as the base model.
    NO CHANGES NEEDED from the Llama/DeiT version.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int, num_layers: int, dtype: torch.dtype):
        super().__init__()
        self.layer_embedding = nn.Embedding(
            num_embeddings=num_layers,
            embedding_dim=hidden_dim
        ).to(device=DEVICE, dtype=dtype)

        self.adapter_network = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim, device=DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim, device=DEVICE, dtype=dtype)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, device=DEVICE, dtype=dtype)

    def forward(self, hidden_state: torch.Tensor, l_curr: int, l_next: int) -> torch.Tensor:
        residual = hidden_state
        # Ensure tensors are created on the correct device from the start
        l_curr_tensor = torch.tensor([l_curr], device=hidden_state.device)
        l_next_tensor = torch.tensor([l_next], device=hidden_state.device)

        curr_layer_emb = self.layer_embedding(l_curr_tensor)
        next_layer_emb = self.layer_embedding(l_next_tensor)

        skip_signal = next_layer_emb - curr_layer_emb

        # The unsqueeze(1) is for broadcasting across the sequence length dimension.
        conditioned_state = hidden_state + skip_signal.unsqueeze(1).to(hidden_state.dtype)
        adapter_output = self.adapter_network(conditioned_state)

        output_state = self.layer_norm(residual + adapter_output)
        return output_state


class ScoringModel(nn.Module):
    """
    A lightweight model to score potential layer skips (our policy network pi_theta).
    It is initialized with the same dtype as the base model.
    NO CHANGES NEEDED from the Llama/DeiT version. The input h_cls (for [CLS] token)
    is a standard feature in both DeiT and BERT.
    """

    def __init__(self, hidden_dim: int, scorer_hidden_dim: int, num_layers: int, dtype: torch.dtype):
        super().__init__()
        embedding_dim_scorer = hidden_dim // 4
        self.layer_embedding = nn.Embedding(
            num_embeddings=num_layers,
            embedding_dim=embedding_dim_scorer
        ).to(device=DEVICE, dtype=dtype)

        self.scorer_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim_scorer * 2, scorer_hidden_dim, device=DEVICE, dtype=dtype),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1, device=DEVICE, dtype=dtype)
        )

    def forward(self, h_cls: torch.Tensor, l_curr: int, candidate_layers: list) -> torch.Tensor:
        batch_size = h_cls.shape[0]
        # Ensure tensors are created on the correct device
        l_curr_tensor = torch.tensor([l_curr] * batch_size, device=h_cls.device)
        scores = []
        for l_cand in candidate_layers:
            l_cand_tensor = torch.tensor([l_cand] * batch_size, device=h_cls.device)
            l_curr_emb = self.layer_embedding(l_curr_tensor)
            l_cand_emb = self.layer_embedding(l_cand_tensor)
            mlp_input = torch.cat([h_cls, l_curr_emb, l_cand_emb], dim=-1)
            score = self.scorer_mlp(mlp_input)
            scores.append(score)
        return torch.cat(scores, dim=-1)
