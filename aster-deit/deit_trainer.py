# /aster-deit/deit_trainer.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

from reward import TIDR_V2


class ImageNetDataset(Dataset):
    """Custom Dataset for loading ImageNet from a CSV file."""

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['filepath']
        label = torch.tensor(row['label'], dtype=torch.long)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class ASTERTrainerDeiT:
    def __init__(self, model, image_transform, scorer, adapter, optimizer, config, rank=0, world_size=1):
        self.model = model
        self.image_transform = image_transform
        self.scorer = scorer
        self.adapter = adapter
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(f'cuda:{rank}')
        self.rank = rank
        self.world_size = world_size

        self.deit_encoder = model.vit.encoder
        self.num_layers = len(self.deit_encoder.layer)

        # --- MODIFICATION: Pass the new skip_penalty_weight to the reward function ---
        self.reward_fn = TIDR_V2(
            w_task=config.W_TASK,
            w_efficiency=config.W_EFFICIENCY,
            total_layers=self.num_layers,
            device=self.device,
            skip_penalty_weight=config.SKIP_PENALTY_WEIGHT
        )

    def _get_final_logits_and_pred(self, final_hidden_state):
        cls_token_state = final_hidden_state[:, 0]
        logits = self.model.classifier(cls_token_state)
        pred_idx = torch.argmax(logits, dim=-1)
        return logits, pred_idx

    def _compute_knowledge_distillation_loss(self, student_state, teacher_state):
        student_cls, student_patches = student_state[:, 0, :], student_state[:, 1:, :]
        teacher_cls, teacher_patches = teacher_state[:, 0, :], teacher_state[:, 1:, :]
        loss_cls = F.mse_loss(student_cls, teacher_cls, reduction='none').mean(dim=1)
        temp = self.config.KD_TEMP
        teacher_attn_scores = torch.bmm(teacher_patches, teacher_cls.unsqueeze(-1)) / temp
        I_teacher = torch.sigmoid(teacher_attn_scores.squeeze(-1))
        student_attn_scores = torch.bmm(student_patches, student_cls.unsqueeze(-1)) / temp
        I_student = torch.sigmoid(student_attn_scores.squeeze(-1))
        importance_weights = (I_teacher * I_student).detach()
        loss_rep_mse = F.mse_loss(student_patches, teacher_patches, reduction='none').mean(dim=2)
        loss_rep_weighted = (importance_weights * loss_rep_mse).mean(dim=1)
        return loss_cls + loss_rep_weighted

    def train(self, train_dataloader, train_sampler, start_epoch=0):
        if self.rank == 0:
            print("--- Starting ASTER-DeiT Training with Individual Policies (Distributed) ---")

        if start_epoch == 0: self.optimizer.zero_grad()

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            if self.rank == 0:
                print(f"\n--- Epoch {epoch + 1}/{self.config.NUM_EPOCHS} ---")

            train_sampler.set_epoch(epoch)

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(self.rank != 0))

            for step, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = images.shape[0]

                with torch.no_grad():
                    teacher_outputs = self.model(images, output_hidden_states=True)
                    teacher_hidden_states = teacher_outputs.hidden_states

                student_hidden_state = teacher_hidden_states[0]

                log_probs_rollout, kd_loss_rollout = [], []
                l_curr_rollout, l_next_rollout = [], []
                l_curr_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                while True:
                    active_mask = l_curr_batch < self.num_layers - 1
                    if not torch.any(active_mask): break

                    l_curr_rollout.append(l_curr_batch.clone())

                    next_hidden_state = student_hidden_state.clone()
                    step_log_probs = torch.zeros(batch_size, device=self.device)
                    step_kd_loss = torch.zeros(batch_size, device=self.device)

                    unique_l_currs = torch.unique(l_curr_batch[active_mask])

                    for l_c in unique_l_currs:
                        l_curr = l_c.item()

                        group_indices = (l_curr_batch == l_curr).nonzero(as_tuple=True)[0]
                        group_hidden_states = student_hidden_state.index_select(0, group_indices)

                        processed_states = self.deit_encoder.layer[l_curr](group_hidden_states)[0]

                        kd_loss = self._compute_knowledge_distillation_loss(
                            processed_states,
                            teacher_hidden_states[l_curr + 1].detach().index_select(0, group_indices)
                        )

                        cls_state = processed_states[:, 0, :]
                        candidate_layers = list(range(l_curr + 1, self.num_layers))

                        if not candidate_layers:
                            l_curr_batch.index_fill_(0, group_indices, self.num_layers)
                            next_hidden_state.index_copy_(0, group_indices, processed_states)
                            continue

                        scores = self.scorer(h_cls=cls_state, l_curr=l_curr, candidate_layers=candidate_layers)
                        probs = F.softmax(scores / 1.5, dim=-1)  # Using a fixed temperature for exploration
                        dist = Categorical(probs=probs)
                        action_indices = dist.sample()
                        log_probs = dist.log_prob(action_indices)

                        l_next_indices = torch.tensor(candidate_layers, device=self.device)[action_indices]

                        final_group_states = processed_states.clone()

                        skip_mask = (l_next_indices > l_curr + 1)
                        if torch.any(skip_mask):
                            unique_l_skips = torch.unique(l_next_indices[skip_mask])
                            for l_skip_choice in unique_l_skips:
                                sub_group_mask = (l_next_indices == l_skip_choice)
                                sub_group_indices_in_group = sub_group_mask.nonzero(as_tuple=True)[0]

                                states_to_adapt = processed_states.index_select(0, sub_group_indices_in_group)
                                adapted_states = self.adapter(
                                    states_to_adapt,
                                    l_curr, l_skip_choice.item()
                                )
                                final_group_states.index_copy_(0, sub_group_indices_in_group, adapted_states)

                        next_hidden_state.index_copy_(0, group_indices, final_group_states)
                        l_curr_batch.index_copy_(0, group_indices, l_next_indices)
                        step_log_probs.index_copy_(0, group_indices, log_probs)
                        step_kd_loss.index_copy_(0, group_indices, kd_loss)

                    student_hidden_state = next_hidden_state
                    log_probs_rollout.append(step_log_probs)
                    kd_loss_rollout.append(step_kd_loss)
                    l_next_rollout.append(l_curr_batch.clone())

                final_logits, pred_idx = self._get_final_logits_and_pred(student_hidden_state)
                final_pred_correct = (pred_idx == labels)

                loss_ce = F.cross_entropy(final_logits, labels)

                num_steps = len(log_probs_rollout)
                if num_steps > 0:
                    rewards = torch.zeros(num_steps, batch_size, device=self.device)
                    for t in range(num_steps):
                        for i in range(batch_size):
                            if l_curr_rollout[t][i] < self.num_layers - 1:
                                is_final = (t == num_steps - 1) or (l_next_rollout[t][i] >= self.num_layers - 1)
                                rewards[t, i] = self.reward_fn.compute_reward(
                                    final_pred_correct[i].item(), l_curr_rollout[t][i].item(),
                                    l_next_rollout[t][i].item(), is_final, t, num_steps
                                )

                    discounted_rewards = torch.zeros_like(rewards)
                    R = torch.zeros(batch_size, device=self.device)
                    for t in reversed(range(num_steps)):
                        R = rewards[t] + self.config.GAMMA * R
                        discounted_rewards[t] = R

                    mean = discounted_rewards.mean(dim=1, keepdim=True)
                    std = discounted_rewards.std(dim=1, keepdim=True) + 1e-8
                    discounted_rewards = (discounted_rewards - mean) / std

                    log_probs_tensor = torch.stack(log_probs_rollout)
                    policy_loss = (-log_probs_tensor * discounted_rewards).sum(dim=0).mean()
                    total_kd_loss = torch.stack(kd_loss_rollout).sum(dim=0).mean()
                else:
                    policy_loss = torch.tensor(0.0, device=self.device)
                    total_kd_loss = torch.tensor(0.0, device=self.device)

                total_loss = (self.config.CE_LOSS_WEIGHT * loss_ce +
                              self.config.RL_LOSS_WEIGHT * policy_loss +
                              self.config.KD_LOSS_WEIGHT * total_kd_loss)

                (total_loss / self.config.GRADIENT_ACCUMULATION_STEPS).backward()

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(list(self.scorer.parameters()) + list(self.adapter.parameters()),
                                                   self.config.CLIP_GRAD_NORM)
                    self.optimizer.step();
                    self.optimizer.zero_grad()

                if self.rank == 0 and (step + 1) % self.config.LOG_INTERVAL == 0:
                    pbar.set_postfix({
                        "Loss": f"{total_loss.item():.4f}",
                        "CE": f"{loss_ce.item():.4f}",
                        "RL": f"{policy_loss.item():.8f}",
                        "KD": f"{total_kd_loss.item():.4f}"})

            if self.rank == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, "aster_deit_checkpoint.pt")

        scorer_state_dict = self.scorer.module.state_dict()
        adapter_state_dict = self.adapter.module.state_dict()

        checkpoint = {'epoch': epoch, 'scorer_state_dict': scorer_state_dict,
                      'adapter_state_dict': adapter_state_dict,
                      'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch + 1} to {checkpoint_path}")