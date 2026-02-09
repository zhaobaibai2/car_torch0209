import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_diffusion import GaussianDiffusion


def _activation(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


def scaled_sinusoidal_encoding(t: torch.Tensor, *, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    t = t.float()
    half_dim = dim // 2
    emb = math.log(theta)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * (-emb / half_dim))
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, *, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.act = _activation(activation)
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU() if activation.lower() == "relu" else nn.GELU() if activation.lower() == "gelu" else nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Sequence[int], *, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.backbone = MLP(obs_dim + act_dim, hidden_dims, 2, activation=activation, use_layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        x = torch.cat([obs, act], dim=-1)
        out = self.backbone(x)
        q_mean = out[..., 0]
        q_std = F.softplus(out[..., 1]) + 0.1
        return q_mean, q_std


class DACERPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Sequence[int], *, time_dim: int = 16, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.time_dim = time_dim
        self.act_fn = _activation(activation)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU() if activation.lower() == "relu" else nn.GELU() if activation.lower() == "gelu" else nn.Tanh(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.mlp = MLP(obs_dim + act_dim + time_dim, hidden_dims, act_dim, activation=activation, use_layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1 and t.shape[0] == 1 and obs.shape[0] > 1:
            t = t.expand(obs.shape[0])

        te = scaled_sinusoidal_encoding(t, dim=self.time_dim)
        te = self.time_mlp(te)
        x = torch.cat([obs, act, te], dim=-1)
        return self.mlp(x)


@dataclass
class DACERActionConfig:
    init_alpha: float = 0.1
    action_noise_scale: float = 0.05


class DACERTorchAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        diffusion_hidden_dims: Sequence[int],
        *,
        num_timesteps: int = 20,
        target_entropy: float = -2.0,
        time_dim: int = 16,
        activation: str = "relu",
        use_layer_norm: bool = True,
        action_cfg: Optional[DACERActionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_timesteps = int(num_timesteps)
        self.target_entropy = float(target_entropy)
        self.device = device or torch.device("cpu")

        self.q1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.target_q1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.target_q2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.policy_net = DACERPolicyNet(self.obs_dim, self.act_dim, diffusion_hidden_dims, time_dim=time_dim, activation=activation, use_layer_norm=use_layer_norm)

        self.diffusion = GaussianDiffusion(self.num_timesteps, device=self.device)

        action_cfg = action_cfg or DACERActionConfig()
        self.action_noise_scale = float(action_cfg.action_noise_scale)
        init_alpha = float(action_cfg.init_alpha)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32, device=self.device))

        self._sync_targets(tau=1.0)

    @torch.no_grad()
    def _sync_targets(self, tau: float = 1.0):
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    @torch.no_grad()
    def soft_update_targets(self, tau: float):
        self._sync_targets(tau=tau)

    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        def model(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.policy_net(obs, x, t)

        if deterministic:
            action = self.diffusion.p_sample_deterministic(model, (obs.shape[0], self.act_dim))
        else:
            action = self.diffusion.p_sample(model, (obs.shape[0], self.act_dim))

        noise = torch.randn_like(action)
        action = action + noise * torch.exp(self.log_alpha) * self.action_noise_scale
        return action.clamp(-1.0, 1.0)

    def q(self, which: int, obs: torch.Tensor, act: torch.Tensor, *, target: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device, dtype=torch.float32)
        act = act.to(self.device, dtype=torch.float32)
        if target:
            net = self.target_q1 if which == 1 else self.target_q2
        else:
            net = self.q1 if which == 1 else self.q2
        return net(obs, act)

    def q_evaluate(self, which: int, obs: torch.Tensor, act: torch.Tensor, *, target: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mean, q_std = self.q(which, obs, act, target=target)
        z = torch.randn_like(q_mean).clamp(-3.0, 3.0)
        q_val = q_mean + q_std * z
        return q_mean, q_std, q_val

    def predict_noise(self, obs: torch.Tensor, t: torch.Tensor, x_noisy: torch.Tensor) -> torch.Tensor:
        return self.policy_net(obs, x_noisy, t)
