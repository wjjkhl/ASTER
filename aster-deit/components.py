# /aster-deit/components.py

import torch
import torch.nn as nn
from config import DEVICE  # Keep this import


class DynamicAdapter(nn.Module):

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
        l_curr_tensor = torch.tensor([l_curr], device=DEVICE)
        l_next_tensor = torch.tensor([l_next], device=DEVICE)

        curr_layer_emb = self.layer_embedding(l_curr_tensor)
        next_layer_emb = self.layer_embedding(l_next_tensor)

        skip_signal = next_layer_emb - curr_layer_emb

        # The original unsqueeze(1) was for sequence data. It works fine here too.
        conditioned_state = hidden_state + skip_signal.unsqueeze(1).to(hidden_state.dtype)
        adapter_output = self.adapter_network(conditioned_state)

        output_state = self.layer_norm(residual + adapter_output)
        return output_state


class ScoringModel(nn.Module):

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
        # Renamed h_cog to h_cls for clarity, but the logic is identical.
        batch_size = h_cls.shape[0]
        l_curr_tensor = torch.tensor([l_curr] * batch_size, device=DEVICE)
        scores = []
        for l_cand in candidate_layers:
            l_cand_tensor = torch.tensor([l_cand] * batch_size, device=DEVICE)
            l_curr_emb = self.layer_embedding(l_curr_tensor)
            l_cand_emb = self.layer_embedding(l_cand_tensor)
            mlp_input = torch.cat([h_cls, l_curr_emb, l_cand_emb], dim=-1)
            score = self.scorer_mlp(mlp_input)
            scores.append(score)

        return torch.cat(scores, dim=-1)
