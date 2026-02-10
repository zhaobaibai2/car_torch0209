from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .torch_networks import Actor, Critic
from .torch_replay_buffer import TorchPVPBuffer, PVPBatch


@dataclass
class TorchPVPTD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 1e-4
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    q_value_bound: float = 1.0
    cql_coefficient: float = 1.0
    reward_free: bool = True
    intervention_start_stop_td: bool = True
    use_balance_sample: bool = True
    human_ratio: float = 0.5


class TorchPVPTD3:
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dims,
        critic_hidden_dims,
        cfg: TorchPVPTD3Config,
        device: torch.device,
    ):
        self.device = device
        self.cfg = cfg

        self.actor = Actor(obs_dim, act_dim, actor_hidden_dims).to(device)
        self.actor_target = Actor(obs_dim, act_dim, actor_hidden_dims).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim, critic_hidden_dims).to(device)
        self.critic_target = Critic(obs_dim, act_dim, critic_hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.step = 0

    @torch.no_grad()
    def predict(self, obs, *, deterministic: bool = True) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        action = self.actor(obs)
        if not deterministic:
            noise = torch.randn_like(action) * self.cfg.target_policy_noise
            action = (action + noise).clamp(-1.0, 1.0)
        action = action.squeeze(0)
        return action

    def train(self, *, buffer: TorchPVPBuffer, batch_size: int, gradient_steps: int) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if gradient_steps <= 0:
            return metrics

        critic_losses = []
        actor_losses = []

        for _ in range(int(gradient_steps)):
            batch = self._sample_batch(buffer, batch_size)
            if batch is None:
                break

            critic_loss = self._update_critic(batch)
            critic_losses.append(critic_loss)

            if self.step % int(self.cfg.policy_delay) == 0:
                actor_loss = self._update_actor(batch)
                actor_losses.append(actor_loss)
                self._soft_update(self.actor, self.actor_target)
                self._soft_update(self.critic, self.critic_target)

            self.step += 1

        if critic_losses:
            metrics["train/critic_loss"] = float(sum(critic_losses) / len(critic_losses))
        if actor_losses:
            metrics["train/actor_loss"] = float(sum(actor_losses) / len(actor_losses))
        return metrics

    def _sample_batch(self, buffer: TorchPVPBuffer, batch_size: int) -> Optional[PVPBatch]:
        if self.cfg.use_balance_sample:
            exps = buffer.sample_pvp_mixed(batch_size, human_ratio=self.cfg.human_ratio)
        else:
            exps = buffer.sample_pvp(batch_size)
        return buffer.to_pvp_batch(exps, device=self.device)

    def _update_critic(self, batch: PVPBatch) -> float:
        with torch.no_grad():
            noise = torch.randn_like(batch.actions_behavior) * self.cfg.target_policy_noise
            noise = noise.clamp(-self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            next_action = (self.actor_target(batch.next_obs) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(batch.next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            backup = batch.reward + (1.0 - batch.done) * self.cfg.gamma * target_q

            if self.cfg.intervention_start_stop_td:
                td_mask = batch.stop_td
            else:
                td_mask = torch.ones_like(batch.stop_td)

            if self.cfg.reward_free:
                td_mask = td_mask * (1.0 - batch.interventions)

        q1_b, q2_b = self.critic(batch.obs, batch.actions_behavior)
        q1_n, q2_n = self.critic(batch.obs, batch.actions_novice)

        td_loss_1 = 0.5 * F.mse_loss(td_mask * q1_b, td_mask * backup)
        td_loss_2 = 0.5 * F.mse_loss(td_mask * q2_b, td_mask * backup)

        cql = float(self.cfg.cql_coefficient)
        bound = float(self.cfg.q_value_bound)
        intervention = batch.interventions

        cql_loss_1 = (intervention * (q1_b - bound) ** 2).mean() + (intervention * (q1_n + bound) ** 2).mean()
        cql_loss_2 = (intervention * (q2_b - bound) ** 2).mean() + (intervention * (q2_n + bound) ** 2).mean()

        critic_loss = td_loss_1 + td_loss_2 + cql * (cql_loss_1 + cql_loss_2)

        self.optim_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optim_critic.step()

        return float(critic_loss.detach().cpu())

    def _update_actor(self, batch: PVPBatch) -> float:
        action = self.actor(batch.obs)
        q1_pi = self.critic.q1_forward(batch.obs, action)
        actor_loss = -q1_pi.mean()

        self.optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim_actor.step()

        return float(actor_loss.detach().cpu())

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        tau = float(self.cfg.tau)
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)

    def save(self, path: str) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "config": asdict(self.cfg),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_target.load_state_dict(payload.get("actor_target", payload["actor"]))
        self.critic_target.load_state_dict(payload.get("critic_target", payload["critic"]))

        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
