# /aster-bert/reward.py
# NO CHANGES NEEDED. This file is algorithm-specific and domain-agnostic.

import torch
import math


class TIDR_V2:
    """
    Implements the advanced TIDR function.
    This version includes a penalty for large layer skips, making it general.
    """

    def __init__(self, w_task, w_efficiency, total_layers, device, skip_penalty_weight=0.0):
        self.w_task = w_task
        self.w_efficiency = w_efficiency
        self.total_layers = total_layers
        self.device = device
        self.skip_penalty_weight = skip_penalty_weight

    def compute_reward(self, final_prediction_correct: bool, l_curr: int, l_next: int,
                       is_final_step: bool, step_t: int, total_steps: int):
        """
        Computes the reward for a single step, including a penalty term.
        The logic is based on universal principles of accuracy and efficiency.
        """
        # --- Time Importance Weight (omega_t) ---
        time_decay = (total_steps - step_t) / total_steps if total_steps > 0 else 0

        if self.total_layers > 0:
            # Use floating point division
            skip_weight_input = torch.tensor((l_next - l_curr) / float(self.total_layers), device=self.device)
            skip_weight = torch.sigmoid(skip_weight_input).item()
        else:
            skip_weight = 0.5

        omega_t = time_decay * skip_weight

        # --- Task Reward (Sparse) ---
        task_reward = 0.0
        if is_final_step:
            task_reward = 1.0 if final_prediction_correct else -1.0

        # --- Efficiency Reward (Logarithmic) ---
        # Add a small epsilon to avoid log(1) = 0 for single-layer steps
        efficiency_reward = math.log(1.001 + l_next - l_curr)

        # --- Skip Penalty Calculation ---
        # A penalty that scales with the square of the normalized skip distance.
        skip_distance = float(l_next - l_curr)
        skip_penalty = (skip_distance / self.total_layers) ** 2 if self.total_layers > 0 else 0.0

        # --- Final Reward Composition with Penalty ---
        total_reward = (self.w_task * task_reward * omega_t) \
                     + (self.w_efficiency * efficiency_reward) \
                     - (self.skip_penalty_weight * skip_penalty)

        return total_reward