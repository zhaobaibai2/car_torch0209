from typing import Iterable, Optional, Sequence, List

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(activation())
        last_dim = dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim, list(hidden_dims), output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.q1 = build_mlp(input_dim, 1, list(hidden_dims))
        self.q2 = build_mlp(input_dim, 1, list(hidden_dims))

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu)
