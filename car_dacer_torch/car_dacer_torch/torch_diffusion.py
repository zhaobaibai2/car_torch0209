import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BetaScheduleCoefficients:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def _vp_beta_schedule(timesteps: int) -> np.ndarray:
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T ** 2))
    betas = 1.0 - alpha
    return betas.astype(np.float32)


def _precompute_coefficients(num_timesteps: int, *, device: torch.device) -> BetaScheduleCoefficients:
    betas = _vp_beta_schedule(num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    return BetaScheduleCoefficients(
        betas=t(betas),
        alphas=t(alphas),
        alphas_cumprod=t(alphas_cumprod),
        alphas_cumprod_prev=t(alphas_cumprod_prev),
        sqrt_alphas_cumprod=t(sqrt_alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=t(sqrt_one_minus_alphas_cumprod),
        log_one_minus_alphas_cumprod=t(log_one_minus_alphas_cumprod),
        sqrt_recip_alphas_cumprod=t(sqrt_recip_alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=t(sqrt_recipm1_alphas_cumprod),
        posterior_variance=t(posterior_variance),
        posterior_log_variance_clipped=t(posterior_log_variance_clipped),
        posterior_mean_coef1=t(posterior_mean_coef1),
        posterior_mean_coef2=t(posterior_mean_coef2),
    )


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 20, *, device: torch.device):
        assert num_timesteps > 0
        self.num_timesteps = int(num_timesteps)
        self.device = device
        self.B = _precompute_coefficients(self.num_timesteps, device=device)

    def p_mean_variance(self, t_index: int, x: torch.Tensor, noise_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        x_recon = x * B.sqrt_recip_alphas_cumprod[t_index] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_index]
        x_recon = x_recon.clamp(-1.0, 1.0)
        model_mean = x_recon * B.posterior_mean_coef1[t_index] + x * B.posterior_mean_coef2[t_index]
        model_log_variance = B.posterior_log_variance_clipped[t_index]
        return model_mean, model_log_variance

    def p_sample(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        noise = torch.randn((self.num_timesteps, *shape), device=self.device, dtype=torch.float32)

        for t in reversed(range(self.num_timesteps)):
            t_vec = torch.full(shape[:-1], t, device=self.device, dtype=torch.long)
            noise_pred = model(t_vec, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            if t > 0:
                x = model_mean + torch.exp(0.5 * model_log_variance) * noise[t] * 0.1
            else:
                x = model_mean
        return x

    @torch.no_grad()
    def p_sample_deterministic(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
        x = torch.zeros(shape, device=self.device, dtype=torch.float32)
        for t in reversed(range(self.num_timesteps)):
            t_vec = torch.full(shape[:-1], t, device=self.device, dtype=torch.long)
            noise_pred = model(t_vec, x)
            model_mean, _ = self.p_mean_variance(t, x, noise_pred)
            x = model_mean
        return x

    def q_sample(self, t: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = self.B
        t = t.long()
        c1 = B.sqrt_alphas_cumprod[t].unsqueeze(-1)
        c2 = B.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return c1 * x_start + c2 * noise

    def p_loss(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        return F.mse_loss(noise_pred, noise)

    def weighted_p_loss(self, weights: torch.Tensor, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        if weights.ndim == 1:
            weights = weights.view(-1, 1)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(t, x_start, noise)
        noise_pred = model(t, x_noisy)

        per_elem = (noise_pred - noise) ** 2
        per_sample = per_elem.mean(dim=-1)
        w = weights.squeeze(-1)
        return (per_sample * w).sum() / (w.sum() + 1e-6)
