from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .torch_networks import DACERTorchAgent
from .torch_replay_buffer import PVPBatch


def compute_pv_loss(
    *,
    q_fn,
    obs: torch.Tensor,
    a_human: torch.Tensor,
    a_novice: torch.Tensor,
    interventions: torch.Tensor,
    B: float,
) -> torch.Tensor:
    I = interventions
    if I.dim() == 2 and I.shape[-1] == 1:
        I = I.squeeze(-1)

    q_h_mean, _ = q_fn(obs, a_human)
    q_n_mean, _ = q_fn(obs, a_novice)

    if q_h_mean.dim() > 1:
        q_h = q_h_mean.squeeze(-1)
    else:
        q_h = q_h_mean

    if q_n_mean.dim() > 1:
        q_n = q_n_mean.squeeze(-1)
    else:
        q_n = q_n_mean

    pv = (q_h - B) ** 2 + (q_n + B) ** 2
    denom = I.sum() + 1e-6
    return (pv * I).sum() / denom


@dataclass
class TorchDACERConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 1e-4
    alpha_lr: float = 3e-2
    delay_update: int = 1
    delay_alpha_update: int = 1000
    reward_scale: float = 1.0
    lambda_pv: float = 1.0
    B: float = 1.0
    lambda_bc: float = 5.0
    reward_free: bool = True
    phase3_use_bc_boost: bool = True
    fix_alpha: bool = True
    target_entropy: float = -2.0


class PVPDACERTorch:
    def __init__(self, agent: DACERTorchAgent, cfg: TorchDACERConfig, *, device: torch.device):
        self.agent = agent
        self.cfg = cfg
        self.device = device

        self.optim_q1 = torch.optim.Adam(self.agent.q1.parameters(), lr=cfg.lr)
        self.optim_q2 = torch.optim.Adam(self.agent.q2.parameters(), lr=cfg.lr)
        self.optim_policy = torch.optim.Adam(self.agent.policy_net.parameters(), lr=cfg.lr)
        self.optim_alpha = torch.optim.Adam([self.agent.log_alpha], lr=cfg.alpha_lr)

        self.step = 0
        self.mean_q1_std = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        self.mean_q2_std = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        self._last_entropy = torch.tensor(0.0, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False, add_noise: bool = True) -> torch.Tensor:
        return self.agent.get_action(obs, deterministic=deterministic, add_noise=add_noise)

    def _update_mean_q_std(self, prev: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if prev.item() < 0.0:
            return new.detach()
        return (self.cfg.tau * new + (1.0 - self.cfg.tau) * prev).detach()

    def _q_loss_distributional(
        self,
        *,
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
        backup_mean_1d: torch.Tensor,
        backup_sample_1d: torch.Tensor,
        mean_q_std: torch.Tensor,
        td_mask_1d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_mean_1d = q_mean.squeeze(-1) if q_mean.dim() > 1 else q_mean
        q_std_1d = q_std.squeeze(-1) if q_std.dim() > 1 else q_std

        q_std_detach = torch.maximum(q_std_1d.detach(), torch.zeros_like(q_std_1d))
        epsilon = 0.1

        q_backup_bounded = (q_mean_1d.detach() + (backup_sample_1d - q_mean_1d.detach()).clamp(-3.0 * mean_q_std, 3.0 * mean_q_std)).detach()

        per_sample_td = -((mean_q_std ** 2 + epsilon) * (
            q_mean_1d * (backup_mean_1d - q_mean_1d).detach() / (q_std_detach ** 2 + epsilon) +
            q_std_1d * (((q_mean_1d.detach() - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon))
        ))

        per_sample_td = per_sample_td * td_mask_1d
        return per_sample_td.mean(), q_mean_1d

    def train_offline_bc(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, float]:
        self.agent.train()
        obs = obs.to(self.device, dtype=torch.float32)
        action = action.to(self.device, dtype=torch.float32)

        t = torch.randint(0, self.agent.num_timesteps, (obs.shape[0],), device=self.device, dtype=torch.long)

        def model(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
            return self.agent.predict_noise(obs, t_batch, x_batch)

        loss = self.agent.diffusion.p_loss(model, t, action)
        loss = self.cfg.lambda_bc * loss

        self.optim_policy.zero_grad(set_to_none=True)
        loss.backward()
        self.optim_policy.step()

        return {'bc_loss': float(loss.detach().cpu())}

    def train_pvp(self, batch: PVPBatch) -> Dict[str, float]:
        self.agent.train()
        self.step += 1

        q_behavior_means = []
        q_novice_means = []

        obs = batch.obs
        next_obs = batch.next_obs
        reward = batch.reward * float(self.cfg.reward_scale)
        done = batch.done
        a_b = batch.actions_behavior
        a_n = batch.actions_novice
        a_h = batch.actions_human
        interventions = batch.interventions
        stop_td = batch.stop_td

        with torch.no_grad():
            next_action = self.agent.get_action(next_obs, deterministic=False)
            next_q1_mean, next_q1_std, next_q1_val = self.agent.q_evaluate(1, next_obs, next_action, target=True)
            next_q2_mean, next_q2_std, next_q2_val = self.agent.q_evaluate(2, next_obs, next_action, target=True)
            next_q_mean = torch.minimum(next_q1_mean, next_q2_mean)
            next_q_sample = torch.where(next_q1_mean < next_q2_mean, next_q1_val, next_q2_val)

            q_target = next_q_mean
            q_target_sample = next_q_sample

            td_mask = (1.0 - stop_td).clamp(0.0, 1.0)

            if bool(self.cfg.reward_free):
                td_mask = td_mask * (1.0 - interventions)

            reward_1d = reward.squeeze(-1) if reward.dim() > 1 else reward
            done_1d = done.squeeze(-1) if done.dim() > 1 else done
            td_mask_1d = td_mask.squeeze(-1) if td_mask.dim() > 1 else td_mask

            q_target_1d = q_target.squeeze(-1) if q_target.dim() > 1 else q_target
            q_target_sample_1d = q_target_sample.squeeze(-1) if q_target_sample.dim() > 1 else q_target_sample

            backup_mean_1d = reward_1d + (1.0 - done_1d) * float(self.cfg.gamma) * q_target_1d
            backup_sample_1d = reward_1d + (1.0 - done_1d) * float(self.cfg.gamma) * q_target_sample_1d

        q1_mean, q1_std = self.agent.q(1, obs, a_b, target=False)
        q2_mean, q2_std = self.agent.q(2, obs, a_b, target=False)

        self.mean_q1_std = self._update_mean_q_std(self.mean_q1_std, q1_std.mean())
        self.mean_q2_std = self._update_mean_q_std(self.mean_q2_std, q2_std.mean())

        q1_td_loss, q1_mean_1d = self._q_loss_distributional(
            q_mean=q1_mean,
            q_std=q1_std,
            backup_mean_1d=backup_mean_1d,
            backup_sample_1d=backup_sample_1d,
            mean_q_std=self.mean_q1_std,
            td_mask_1d=td_mask_1d,
        )
        q2_td_loss, q2_mean_1d = self._q_loss_distributional(
            q_mean=q2_mean,
            q_std=q2_std,
            backup_mean_1d=backup_mean_1d,
            backup_sample_1d=backup_sample_1d,
            mean_q_std=self.mean_q2_std,
            td_mask_1d=td_mask_1d,
        )

        pv1 = compute_pv_loss(q_fn=lambda o, a: self.agent.q(1, o, a, target=False), obs=obs, a_human=a_h, a_novice=a_n, interventions=interventions, B=float(self.cfg.B))
        pv2 = compute_pv_loss(q_fn=lambda o, a: self.agent.q(2, o, a, target=False), obs=obs, a_human=a_h, a_novice=a_n, interventions=interventions, B=float(self.cfg.B))
        pv_loss = 0.5 * (pv1 + pv2)

        q1_loss = q1_td_loss + float(self.cfg.lambda_pv) * pv1
        q2_loss = q2_td_loss + float(self.cfg.lambda_pv) * pv2

        self.optim_q1.zero_grad(set_to_none=True)
        q1_loss.backward(retain_graph=True)
        self.optim_q1.step()

        self.optim_q2.zero_grad(set_to_none=True)
        q2_loss.backward(retain_graph=True)
        self.optim_q2.step()

        policy_loss = torch.tensor(0.0, device=self.device)
        rl_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)

        q_behavior_means.append(float(q1_mean.mean().detach().cpu()))
        q_novice_means.append(float(q2_mean.mean().detach().cpu()))

        if self.step % int(self.cfg.delay_update) == 0:
            for p in self.agent.q1.parameters():
                p.requires_grad_(False)
            for p in self.agent.q2.parameters():
                p.requires_grad_(False)

            new_action = self.agent.get_action(obs, deterministic=False)
            q1_pi, _ = self.agent.q(1, obs, new_action, target=False)
            q2_pi, _ = self.agent.q(2, obs, new_action, target=False)
            q_pi = torch.minimum(q1_pi, q2_pi)

            I = interventions
            if I.dim() == 2 and I.shape[-1] == 1:
                I = I.squeeze(-1)
            notI = 1.0 - I
            q_scalar = q_pi.squeeze(-1) if q_pi.dim() > 1 else q_pi
            rl_loss = (-q_scalar * notI).sum() / (notI.sum() + 1e-6)

            if bool(self.cfg.phase3_use_bc_boost):
                weights = interventions
                t = torch.randint(0, self.agent.num_timesteps, (obs.shape[0],), device=self.device, dtype=torch.long)

                def model(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
                    return self.agent.predict_noise(obs, t_batch, x_batch)

                bc_loss = self.agent.diffusion.weighted_p_loss(weights, model, t, a_h)
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            policy_loss = rl_loss + float(self.cfg.lambda_bc) * bc_loss

            self.optim_policy.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.optim_policy.step()

            for p in self.agent.q1.parameters():
                p.requires_grad_(True)
            for p in self.agent.q2.parameters():
                p.requires_grad_(True)

            with torch.no_grad():
                self.agent.soft_update_targets(float(self.cfg.tau))

        alpha_loss = torch.tensor(0.0, device=self.device)
        if not bool(self.cfg.fix_alpha) and (self.step % int(self.cfg.delay_alpha_update) == 0) and self.step > 0:
            alpha = torch.exp(self.agent.log_alpha)
            alpha_loss = -(self.agent.log_alpha * (-self._last_entropy.detach() + float(self.cfg.target_entropy))).mean()
            self.optim_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
        return metrics

    def save(self, path: str) -> None:
        data = {
            'agent': self.agent.state_dict(),
            'optim_q1': self.optim_q1.state_dict(),
            'optim_q2': self.optim_q2.state_dict(),
            'optim_policy': self.optim_policy.state_dict(),
            'optim_alpha': self.optim_alpha.state_dict(),
            'step': self.step,
            'mean_q1_std': float(self.mean_q1_std.detach().cpu()),
            'mean_q2_std': float(self.mean_q2_std.detach().cpu()),
        }
        torch.save(data, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(data['agent'])
        self.optim_q1.load_state_dict(data.get('optim_q1', {}))
        self.optim_q2.load_state_dict(data.get('optim_q2', {}))
        self.optim_policy.load_state_dict(data.get('optim_policy', {}))
        self.optim_alpha.load_state_dict(data.get('optim_alpha', {}))
        self.step = int(data.get('step', 0))
        self.mean_q1_std = torch.tensor(float(data.get('mean_q1_std', -1.0)), device=self.device)
        self.mean_q2_std = torch.tensor(float(data.get('mean_q2_std', -1.0)), device=self.device)
