from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional
from collections import deque
import pickle

import numpy as np
import torch


@dataclass
class Experience:
    obs: np.ndarray
    next_obs: np.ndarray
    reward: float
    done: bool
    actions_behavior: np.ndarray
    actions_novice: np.ndarray
    actions_human: np.ndarray
    interventions: float
    stop_td: float

    @staticmethod
    def from_pvp(
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        a_behavior: np.ndarray,
        a_novice: np.ndarray,
        a_human: np.ndarray,
        intervention: float,
        stop_td: float,
    ) -> "Experience":
        return Experience(
            obs=np.asarray(obs, dtype=np.float32),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            reward=float(reward),
            done=bool(done),
            actions_behavior=np.asarray(a_behavior, dtype=np.float32),
            actions_novice=np.asarray(a_novice, dtype=np.float32),
            actions_human=np.asarray(a_human, dtype=np.float32),
            interventions=float(intervention),
            stop_td=float(stop_td),
        )


class PVPBatch(NamedTuple):
    obs: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    reward: torch.Tensor
    action: torch.Tensor
    actions_behavior: torch.Tensor
    actions_novice: torch.Tensor
    actions_human: torch.Tensor
    interventions: torch.Tensor
    stop_td: torch.Tensor


class TorchPVPBuffer:
    def __init__(self, *, max_size: int, obs_dim: int, act_dim: int):
        self.max_size = int(max_size)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.human_buffer = deque(maxlen=max_size // 2)
        self.pvp_buffer = deque(maxlen=max_size)

        self.total_samples = 0
        self.total_interventions = 0

    @property
    def human_size(self) -> int:
        return len(self.human_buffer)

    @property
    def pvp_size(self) -> int:
        return len(self.pvp_buffer)

    def add_human(self, exp: Experience) -> None:
        self.human_buffer.append(exp)
        self.total_samples += 1
        if exp.interventions > 0.5:
            self.total_interventions += 1

    def add_pvp(self, exp: Experience) -> None:
        self.pvp_buffer.append(exp)
        self.total_samples += 1
        if exp.interventions > 0.5:
            self.total_interventions += 1

    def sample_human(self, batch_size: int) -> List[Experience]:
        if self.human_size == 0:
            return []
        n = min(int(batch_size), self.human_size)
        idx = np.random.choice(self.human_size, n, replace=False)
        return [self.human_buffer[i] for i in idx]

    def sample_pvp(self, batch_size: int) -> List[Experience]:
        if self.pvp_size == 0:
            return []
        n = min(int(batch_size), self.pvp_size)
        idx = np.random.choice(self.pvp_size, n, replace=False)
        return [self.pvp_buffer[i] for i in idx]

    def sample_pvp_mixed(self, batch_size: int, *, human_ratio: float) -> List[Experience]:
        batch_size = int(batch_size)
        human_target = int(batch_size * float(human_ratio))
        human_exps = self.sample_human(human_target) if human_target > 0 else []
        pvp_target = batch_size - len(human_exps)
        pvp_exps = self.sample_pvp(pvp_target) if pvp_target > 0 else []
        return human_exps + pvp_exps

    def to_pvp_batch(self, exps: List[Experience], *, device: torch.device) -> Optional[PVPBatch]:
        if not exps:
            return None

        obs = torch.as_tensor(np.stack([e.obs for e in exps], axis=0), device=device, dtype=torch.float32)
        next_obs = torch.as_tensor(np.stack([e.next_obs for e in exps], axis=0), device=device, dtype=torch.float32)
        reward = torch.as_tensor(np.asarray([e.reward for e in exps], dtype=np.float32).reshape(-1, 1), device=device)
        done = torch.as_tensor(np.asarray([e.done for e in exps], dtype=np.float32).reshape(-1, 1), device=device)

        a_b = torch.as_tensor(np.stack([e.actions_behavior for e in exps], axis=0), device=device, dtype=torch.float32)
        a_n = torch.as_tensor(np.stack([e.actions_novice for e in exps], axis=0), device=device, dtype=torch.float32)
        a_h = torch.as_tensor(np.stack([e.actions_human for e in exps], axis=0), device=device, dtype=torch.float32)

        interventions = torch.as_tensor(np.asarray([e.interventions for e in exps], dtype=np.float32).reshape(-1, 1), device=device)
        stop_td = torch.as_tensor(np.asarray([e.stop_td for e in exps], dtype=np.float32).reshape(-1, 1), device=device)

        return PVPBatch(
            obs=obs,
            next_obs=next_obs,
            done=done,
            reward=reward,
            action=a_b,
            actions_behavior=a_b,
            actions_novice=a_n,
            actions_human=a_h,
            interventions=interventions,
            stop_td=stop_td,
        )

    def get_statistics(self) -> Dict[str, int]:
        return {
            'human_size': self.human_size,
            'pvp_size': self.pvp_size,
            'total_samples': self.total_samples,
            'total_interventions': self.total_interventions,
        }

    def save(self, path: str) -> None:
        data = {
            'human_buffer': list(self.human_buffer),
            'pvp_buffer': list(self.pvp_buffer),
            'total_samples': self.total_samples,
            'total_interventions': self.total_interventions,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.human_buffer = deque(data.get('human_buffer', []), maxlen=self.max_size // 2)
        self.pvp_buffer = deque(data.get('pvp_buffer', []), maxlen=self.max_size)
        self.total_samples = int(data.get('total_samples', 0))
        self.total_interventions = int(data.get('total_interventions', 0))
