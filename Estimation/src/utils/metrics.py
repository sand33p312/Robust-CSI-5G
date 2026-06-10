"""
utils/metrics.py
Loss functions and NMSE metrics shared by training and evaluation.
"""

import numpy as np
import torch


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL = 0.5 * mean[(y-μ)² * exp(-lv) + lv]."""
    return 0.5 * torch.mean((target - mean) ** 2 * torch.exp(-log_var) + log_var)


def elbo_loss(model, x: torch.Tensor, h: torch.Tensor,
              kl_w: float, mc: int):
    """
    ELBO = mean(NLL over mc samples) + kl_w * KL

    Returns: total_loss, nll (scalar), kl (scalar)
    """
    nll_sum = sum(gaussian_nll(*model(x, True), h) for _ in range(mc)) / mc
    kl = model.kl_divergence()
    return nll_sum + kl_w * kl, nll_sum.item(), kl.item()


def nmse_db_global(preds: torch.Tensor, trues: torch.Tensor,
                   h_sigs: torch.Tensor) -> float:
    """
    Global NMSE in dB, computed in physical (denormalised) domain.

      NMSE = 10 * log10( Σ||ĥ - h||² / Σ||h||² )

    h_sigs: per-sample normalisation std (from SmartCDLDataset)
    """
    P = preds * h_sigs.unsqueeze(-1)
    T = trues * h_sigs.unsqueeze(-1)
    return (10 * torch.log10(
        ((P - T) ** 2).sum() / ((T ** 2).sum() + 1e-12)
    )).item()


def nmse_db_np(pred: np.ndarray, true: np.ndarray) -> float:
    """NumPy version (physical domain, already denormalised)."""
    return 10 * np.log10(
        ((pred - true) ** 2).sum() / ((true ** 2).sum() + 1e-12)
    )