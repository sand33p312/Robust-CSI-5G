"""
model/bnn_layers.py
Bayesian neural network building blocks (mean-field variational inference).

Hierarchy:
  VIModule (base)
    └─ BayesLinear       — learnable Gaussian w/ reparameterisation
    └─ BayesConv2d       — same but conv filters
    └─ BayesResBlock     — 2× BayesConv2d + GroupNorm + skip
    └─ MultiScaleBranch  — parallel 3×3 / 1×3 / 3×1 → fuse with 1×1

Key design decisions:
  - KL collected via .modules() (covers BayesLinear inside ModuleList)
  - GroupNorm throughout (BatchNorm breaks under MC sampling)
  - Weight samples stored via standard nn.Parameter (no __setattr__ bypass needed
    here — sampled values stay local to forward())
  - INIT_SIGMA=0.05: start meaningfully stochastic; collapse is prevented by
    the fixed KL weight rather than adaptive scheduling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn_groups(nc: int) -> int:
    """Largest divisor of nc that is ≤ 8 and divides evenly.
    Used by GroupNorm to stay compatible with any channel count."""
    return max(d for d in [1, 2, 3, 4, 6, 8] if nc % d == 0)


# ── Base VI module ─────────────────────────────────────────────────────────────

class VIModule(nn.Module):
    """Base class for variational inference modules.

    Every subclass calls self.addLoss(fn) with a closure that computes its
    own KL contribution.  kl_divergence() recursively sums all registered
    losses across the full module tree.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._losses = []

    def addLoss(self, fn):
        self._losses.append(fn)

    def evalLosses(self):
        return sum(l(self) for l in self._losses)

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL across every VIModule in the tree (uses .modules() — NOT .children())."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, VIModule) and len(m._losses) > 0:
                total = total + m.evalLosses()
        return total


# ── Bayesian layers ────────────────────────────────────────────────────────────

class BayesLinear(VIModule):
    """Fully-connected Bayesian layer.

    Parameters are (weight_mean, log_weight_sigma, bias_mean, log_bias_sigma).
    Forward samples w ~ N(wm, exp(lws)²) when sample=True, else uses mean.
    KL vs N(0, ps²) prior added to self._losses.
    """

    def __init__(self, in_f: int, out_f: int, ps: float = 1.0, iss: float = 0.05):
        super().__init__()
        self.wm  = nn.Parameter(torch.empty(out_f, in_f))
        self.lws = nn.Parameter(torch.full((out_f, in_f), math.log(iss)))
        self.bm  = nn.Parameter(torch.empty(out_f))
        self.lbs = nn.Parameter(torch.full((out_f,), math.log(iss)))

        nn.init.kaiming_uniform_(self.wm, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wm)
        b = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bm, -b, b)

        p2 = ps ** 2
        self.addLoss(lambda s: 0.5 * torch.sum(
            (s.wm**2 + torch.exp(2*s.lws)) / p2 - 1 - 2*s.lws + 2*math.log(ps)))
        self.addLoss(lambda s: 0.5 * torch.sum(
            (s.bm**2 + torch.exp(2*s.lbs)) / p2 - 1 - 2*s.lbs + 2*math.log(ps)))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        ws = torch.exp(self.lws)
        bs = torch.exp(self.lbs)
        w = self.wm + (ws * torch.randn_like(ws) if sample else 0)
        b = self.bm + (bs * torch.randn_like(bs) if sample else 0)
        return F.linear(x, w, b)


class BayesConv2d(VIModule):
    """2-D Bayesian convolution.

    Identical logic to BayesLinear but over conv filters.
    """

    def __init__(self, ic: int, oc: int, k, stride: int = 1,
                 padding: int = 0, bias: bool = True,
                 ps: float = 1.0, iss: float = 0.05):
        super().__init__()
        self.stride = stride; self.padding = padding; self.has_bias = bias
        ks = (k, k) if isinstance(k, int) else k

        self.wm  = nn.Parameter(torch.empty(oc, ic, *ks))
        self.lws = nn.Parameter(torch.full((oc, ic, *ks), math.log(iss)))
        nn.init.kaiming_uniform_(self.wm, a=math.sqrt(5))

        p2 = ps ** 2
        self.addLoss(lambda s: 0.5 * torch.sum(
            (s.wm**2 + torch.exp(2*s.lws)) / p2 - 1 - 2*s.lws + 2*math.log(ps)))

        if bias:
            self.bm  = nn.Parameter(torch.zeros(oc))
            self.lbs = nn.Parameter(torch.full((oc,), math.log(iss)))
            self.addLoss(lambda s: 0.5 * torch.sum(
                (s.bm**2 + torch.exp(2*s.lbs)) / p2 - 1 - 2*s.lbs + 2*math.log(ps)))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        ws = torch.exp(self.lws)
        w = self.wm + (ws * torch.randn_like(ws) if sample else 0)
        b = None
        if self.has_bias:
            bs = torch.exp(self.lbs)
            b = self.bm + (bs * torch.randn_like(bs) if sample else 0)
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


# ── Composite blocks ──────────────────────────────────────────────────────────

class BayesResBlock(VIModule):
    """Residual block with two BayesConv2d layers and GroupNorm."""

    def __init__(self, ic: int, oc: int, ps: float = 1.0, iss: float = 0.05):
        super().__init__()
        self.c1 = BayesConv2d(ic, oc, 3, padding=1, bias=False, ps=ps, iss=iss)
        self.g1 = nn.GroupNorm(gn_groups(oc), oc)
        self.c2 = BayesConv2d(oc, oc, 3, padding=1, bias=False, ps=ps, iss=iss)
        self.g2 = nn.GroupNorm(gn_groups(oc), oc)
        self._skip = (ic != oc)
        if self._skip:
            self.sc = BayesConv2d(ic, oc, 1, bias=False, ps=ps, iss=iss)
            self.gs = nn.GroupNorm(gn_groups(oc), oc)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        res = self.gs(self.sc(x, sample)) if self._skip else x
        out = F.relu(self.g1(self.c1(x, sample)))
        out = self.g2(self.c2(out, sample))
        return F.relu(out + res)


class MultiScaleBranch(VIModule):
    """
    Three parallel branches — different kernel shapes — then fused with 1×1 BayesConv.

      Branch A: 3×3 — captures joint frequency-time patterns
      Branch B: 1×3 — frequency-selective (across SCs)
      Branch C: 3×1 — time-selective (across OFDM symbols)

    Motivated by the non-symmetric nature of 5G NR resource grids
    (12 SC × 14 sym).
    """

    def __init__(self, in_ch: int, out_ch: int, ps: float = 1.0, iss: float = 0.05):
        super().__init__()
        self.brA = BayesConv2d(in_ch, out_ch, (3, 3), padding=(1, 1), bias=False, ps=ps, iss=iss)
        self.gnA = nn.GroupNorm(gn_groups(out_ch), out_ch)
        self.brB = BayesConv2d(in_ch, out_ch, (1, 3), padding=(0, 1), bias=False, ps=ps, iss=iss)
        self.gnB = nn.GroupNorm(gn_groups(out_ch), out_ch)
        self.brC = BayesConv2d(in_ch, out_ch, (3, 1), padding=(1, 0), bias=False, ps=ps, iss=iss)
        self.gnC = nn.GroupNorm(gn_groups(out_ch), out_ch)
        self.fuse = BayesConv2d(3 * out_ch, out_ch, 1, bias=False, ps=ps, iss=iss)
        self.gnF  = nn.GroupNorm(gn_groups(out_ch), out_ch)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        a = F.relu(self.gnA(self.brA(x, sample)))
        b = F.relu(self.gnB(self.brB(x, sample)))
        c = F.relu(self.gnC(self.brC(x, sample)))
        return F.relu(self.gnF(self.fuse(torch.cat([a, b, c], dim=1), sample)))