"""
utils/metrics.py
Loss functions and NMSE metrics shared by training and evaluation.
"""

import numpy as np
import torch


def gaussian_nll(mean, log_var, target):
    """Heteroscedastic Gaussian NLL = 0.5 * mean[(y-mu)^2 * exp(-lv) + lv]."""
    return 0.5 * torch.mean((target - mean) ** 2 * torch.exp(-log_var) + log_var)


def elbo_loss(model, x, h, kl_w, mc):
    """
    ELBO = mean(NLL over mc samples) + kl_w * KL

    Returns: total_loss, nll (scalar), kl (scalar)
    """
    nll_sum = sum(gaussian_nll(*model(x, True), h) for _ in range(mc)) / mc
    kl = model.kl_divergence()
    return nll_sum + kl_w * kl, nll_sum.item(), kl.item()


def nmse_db_global(preds, trues, scales):
    """
    Global NMSE in dB, computed in the RAW physical domain.

      NMSE = 10 * log10( sum||H_hat - H||^2 / sum||H||^2 )

    `scales` is the per-sample shared normalisation scale produced by
    sample_to_real(). It multiplies BOTH prediction and truth, so it cancels in
    the ratio - the value is identical with or without denormalisation. It is
    kept for explicitness and so the physical-domain arrays are available for
    downstream analysis (calibration, retention, OOD).
    """
    P = preds * scales.unsqueeze(-1)
    T = trues * scales.unsqueeze(-1)
    return (10 * torch.log10(
        ((P - T) ** 2).sum() / ((T ** 2).sum() + 1e-12)
    )).item()


def nmse_db_np(pred, true):
    """NumPy version (physical domain, already denormalised)."""
    return 10 * np.log10(
        ((pred - true) ** 2).sum() / ((true ** 2).sum() + 1e-12)
    )