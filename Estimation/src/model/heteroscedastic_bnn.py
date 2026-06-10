"""
model/heteroscedastic_bnn.py
HeteroscedasticBNN v4 — Hybrid multi-scale encoder with MLP mean decoder
and spatial CNN logvar decoder.

Architecture summary:

  Input [B, 10, 12, 14]   (2*NTX + 2*NRX channels, NSC×NSYM grid)
      ↓
  Shared encoder:
    BayesConv stem (10→32) + GN + ReLU
    MultiScaleBranch (32→32): parallel 3×3 | 1×3 | 3×1 kernels
    BayesResBlock (32→64)
    BayesResBlock (64→64)
    → f  [B, 64, 12, 14]
      ↓
  ┌───────────────────────┬─────────────────────────────────────────┐
  │  Mean branch (MLP)    │  Logvar branch (spatial CNN)            │
  │  f → flatten 10752    │  f * α + f.detach() * (1-α)  (α=0.1)   │
  │  BayesLinear 10752→hd │  Conv2d(64→32) + GN + ReLU             │
  │  LayerNorm + ReLU     │  Conv2d(32→12) + GN + ReLU             │
  │  BayesLinear hd→hd    │  Conv2d(12→12, 1×1) per-RE logvar       │
  │  LayerNorm + ReLU     │  clamp(−10, 4)  → [B,12,12,14]          │
  │  BayesLinear hd→2016  │  flatten → [B, 2016]                    │
  │  → mean [B, 2016]     │  → log_var [B, 2016]                    │
  └───────────────────────┴─────────────────────────────────────────┘

Why hybrid (mean MLP + logvar CNN):
  Pure CNN decoder gave -25.9 dB ceiling (1×1 conv output = 768 params for 2016 outputs).
  MLP decoder (v3) gave -39.77 dB (3.3M params in decoder).
  Fix: keep MLP for mean prediction; CNN per-RE spatial decoder for logvar only.

Logvar changes vs v3:
  - α=0.1 gradient leak (instead of full detach): lets aleatoric head
    see a small gradient signal from the encoder, improving per-sample discrimination
  - clamp(-10, 4) instead of (-6, 4): removes the floor that was pinning
    aleatoric at exp(-6)=0.00248 above 10 dB
  - lv_out.bias init = -2.0: initial σ ≈ exp(-2/2)=0.37 (closer to noise scale)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnn_layers import VIModule, BayesLinear, BayesConv2d, BayesResBlock, MultiScaleBranch, gn_groups


class HeteroscedasticBNN(VIModule):

    def __init__(
        self,
        cin:     int   = 10,
        cch:     int   = 64,
        hd:      int   = 256,
        nh:      int   = 2,
        nsc:     int   = 12,
        nsym:    int   = 14,
        nch_out: int   = 12,
        od:      int   = 2016,
        ps:      float = 1.0,
        iss:     float = 0.05,
        alpha:   float = 0.1,     # logvar gradient leak coefficient
    ):
        super().__init__()
        self.nsc     = nsc
        self.nsym    = nsym
        self.nch_out = nch_out
        self.alpha   = alpha
        flat_dim     = cch * nsc * nsym  # 64*12*14 = 10752

        # ── Shared encoder ──────────────────────────────────────────────────
        self.stem    = BayesConv2d(cin, 32, 3, padding=1, bias=False, ps=ps, iss=iss)
        self.stem_gn = nn.GroupNorm(gn_groups(32), 32)
        self.ms      = MultiScaleBranch(32, 32, ps, iss)
        self.r1      = BayesResBlock(32, cch, ps, iss)
        self.r2      = BayesResBlock(cch, cch, ps, iss)

        # ── Mean branch: Bayesian MLP ────────────────────────────────────────
        dims     = [flat_dim] + [hd] * nh + [od]
        self.fc  = nn.ModuleList([BayesLinear(dims[i], dims[i+1], ps, iss)
                                  for i in range(len(dims) - 1)])
        self.lns = nn.ModuleList([nn.LayerNorm(hd) for _ in range(nh)])

        # ── Logvar branch: deterministic spatial CNN ─────────────────────────
        self.lv_r1  = nn.Sequential(
            nn.Conv2d(cch,   32,      3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(32), 32),
            nn.ReLU(),
        )
        self.lv_r2  = nn.Sequential(
            nn.Conv2d(32,    nch_out, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(nch_out), nch_out),
            nn.ReLU(),
        )
        self.lv_out = nn.Conv2d(nch_out, nch_out, 1)  # per-RE logvar
        nn.init.zeros_(self.lv_out.weight)
        nn.init.constant_(self.lv_out.bias, -2.0)     # init σ ≈ 0.37

        # Parameter count summary
        n      = sum(p.numel() for p in self.parameters())
        n_lv   = sum(p.numel() for nm, p in self.named_parameters() if 'lv_' in nm)
        print(f'  HeteroscedasticBNN v4 (hybrid): {n:,} params')
        print(f'    Encoder (shared): ~{n-n_lv:,}  |  Logvar branch: {n_lv:,}')
        print(f'    Mean: MLP {flat_dim}→{hd}→{hd}→{od}')
        print(f'    Logvar: CNN {cch}→32→{nch_out}→{nch_out} | α={alpha} | clamp(-10,4)')

    def forward(self, x: torch.Tensor, sample: bool = True):
        """
        Args:
          x       [B, CNN_IN_CH, NSC, NSYM]
          sample  if True, sample Bayesian weights; if False, use means
                  (use False for ensemble member forward pass)
        Returns:
          mean    [B, OUTPUT_DIM]
          log_var [B, OUTPUT_DIM]
        """
        # ── Shared encoder ─────────────────────────────────────────────────
        f = F.relu(self.stem_gn(self.stem(x, sample)))   # [B, 32, 12, 14]
        f = self.ms(f, sample)                            # [B, 32, 12, 14]
        f = self.r1(f, sample)                            # [B, 64, 12, 14]
        f = self.r2(f, sample)                            # [B, 64, 12, 14]

        # ── Mean branch ─────────────────────────────────────────────────────
        h = f.flatten(1)                                  # [B, 10752]
        for i, layer in enumerate(self.fc[:-1]):
            h = F.relu(self.lns[i](layer(h, sample)))    # [B, hd]
        mean = self.fc[-1](h, sample)                     # [B, 2016]

        # ── Logvar branch ────────────────────────────────────────────────────
        # α=0.1 gradient leak: small encoder gradient reaches logvar head,
        # enabling per-sample difficulty discrimination
        f_leaked = f * self.alpha + f.detach() * (1.0 - self.alpha)
        lv = self.lv_r1(f_leaked)                         # [B, 32, 12, 14]
        lv = self.lv_r2(lv)                               # [B, 12, 12, 14]
        lv = self.lv_out(lv)                              # [B, 12, 12, 14]
        lv = lv.clamp(-10, 4)                             # σ ∈ [exp(-5), exp(2)]
        log_var = lv.flatten(1)                           # [B, 2016]

        return mean, log_var


def build_model(cfg) -> HeteroscedasticBNN:
    """Convenience factory: build model from config module."""
    return HeteroscedasticBNN(
        cin     = cfg.CNN_IN_CH,
        cch     = cfg.CNN_CHANNELS,
        hd      = cfg.HIDDEN_DIM,
        nh      = cfg.NUM_HIDDEN,
        nsc     = cfg.NSC,
        nsym    = cfg.NSYM,
        nch_out = cfg.NCH_OUT,
        od      = cfg.OUTPUT_DIM,
        ps      = cfg.PRIOR_SIGMA,
        iss     = cfg.INIT_SIGMA,
    ).to(cfg.DEVICE)