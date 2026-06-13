# Phase 1 — Channel Estimation with Bayesian Neural Networks

**Task**: Reconstruct full CSI `H` from sparse pilot observations `Y` for 5G NR CDL channels.
**Config**: MIMO 3×2 (3 Tx × 2 Rx) · 3GPP CDL-A/C/D train · CDL-B/E OOD · 3.5 GHz · 12 SC × 14 sym
**Method**: Deep Ensemble (K=5) of Heteroscedastic BNN members with calibrated uncertainty

---

## Results (Deep Ensemble, K=5, corrected normalisation)

> **Normalisation note.** Earlier numbers (NMSE −40.28 dB) used a target normalised by its
> own ground-truth std (`h_sig`), which coupled the learning target to statistics the receiver
> never has at deployment and inflated NMSE by ~5 dB. The pipeline now uses a **single shared
> physical scale** for input and target (see Dataset section). The values below are the
> corrected, deployment-honest, CeBed/Channelformer-comparable numbers — and the correction
> made NMSE honest *and* improved every uncertainty metric.

### NMSE vs SNR

| SNR | Ensemble NMSE |
|---|---|
| −5 dB | −18.09 dB |
| 0 dB | −20.61 dB |
| 5 dB | −24.17 dB |
| 10 dB | −28.37 dB |
| 15 dB | −32.71 dB |
| 20 dB | **−36.16 dB** |
| 25 dB | −38.16 dB |
| 30 dB | −38.80 dB |
| Global | −24.18 dB |

Best single member @ 20 dB: −35.83 dB → ensemble averaging adds **+0.33 dB** (and +1.88 dB @ 30 dB over the single-BNN baseline).

### Uncertainty & OOD

| Metric | Value |
|---|---|
| Per-SNR Pearson r (total σ) | **0.9815** |
| All-sample Pearson r (total σ) | 0.6925 |
| Calibrated σ within-SNR r | **0.4537** |
| Epistemic within-SNR r (raw) | +0.298 (peak +0.402 @ 10 dB) |
| Component r — aleatoric / epistemic | 0.693 / 0.556 |
| OOD AUROC vs CDL-B/E (pooled) | **0.9933** (1.0000 at SNR ≥ 15 dB) |
| OOD epistemic elevation (CDL-B/E) | **66.4×** |
| OOD AUROC vs extreme Dop/DS (graded) | 0.7807 (0.61 → 0.91 with SNR) |
| Mixed-pool: OOD share of discarded 50% | **100.0%** |
| AUSE (σ_cal / epis) | 2.96 / 3.20 dB·frac |

**The contribution is not lower NMSE — it is calibrated uncertainty.** Every published baseline
(ChannelNet, ReEsNet, Channelformer, HELENA, ReQuestNet, CEHNet, …) is a point estimator. None
produce per-prediction epistemic/aleatoric uncertainty, OOD detection via ensemble disagreement,
or selective-prediction retention curves. This work does.

---

## Architecture — Heteroscedastic BNN (Hybrid)

```
Input [B, 10, 12, 14]   ← 2*NTX + 2*NRX channels, NSC × NSYM grid
        ↓
Shared Encoder:
  BayesConv stem (10→32) + GN + ReLU
  MultiScaleBranch (32→32): parallel 3×3 | 1×3 | 3×1 kernels
  BayesResBlock (32→64)
  BayesResBlock (64→64)
  → f [B, 64, 12, 14]
        ↓
  ┌─────────────────────┬────────────────────────────────┐
  │  Mean branch (MLP)  │  Logvar branch (spatial CNN)   │
  │  flatten → 10752    │  f * α + f.detach() * (1-α)    │
  │  BayesLinear→256    │  Conv2d 64→32→12→12 (1×1)      │
  │  LayerNorm + ReLU   │  clamp(−10, 4)                 │
  │  BayesLinear→256    │  → log_var [B, 2016]           │
  │  BayesLinear→2016   │                                │
  │  → mean [B, 2016]   │                                │
  └─────────────────────┴────────────────────────────────┘
```

**Key design choices**
- **MLP decoder for mean**: 3.3M params reaches the −38 dB regime. Pure CNN decoder hits a −25.9 dB ceiling (1×1 output bottleneck).
- **α=0.1 gradient leak**: lets the aleatoric head see encoder gradients → per-sample discrimination.
- **clamp(−10, 4)**: removes the −6 floor that pinned σ² at high SNR.
- **GroupNorm throughout**: BatchNorm breaks under MC weight sampling.
- **Fixed KL weight**: adaptive beta creates positive feedback loops.
- **Deep ensemble (K=5)**: epistemic = member disagreement `var_k μ_k(x)` — input-dependent function-space uncertainty that mean-field VI structurally cannot provide.

---

## Code Structure

```
Estimation/
├── README.md
└── src/
    ├── config.py                    ← all hyperparameters and paths
    ├── dataset/
    │   ├── cdl_dataset.py           ← SmartCDLDataset_Train/Eval + build_loaders()
    │   └── __init__.py
    ├── model/
    │   ├── bnn_layers.py            ← VIModule, BayesLinear/Conv2d, BayesResBlock, MultiScaleBranch
    │   ├── heteroscedastic_bnn.py   ← HeteroscedasticBNN + build_model()
    │   └── __init__.py
    ├── train/
    │   ├── train_member.py          ← train one ensemble member (--seed 0..4)
    │   └── run_all_members.sh       ← runs all 5 seeds sequentially
    ├── evaluate/
    │   └── ensemble_eval.py         ← NMSE + calibration + OOD + retention + AUROC + plots
    └── utils/
        ├── metrics.py               ← gaussian_nll, elbo_loss, nmse_db_global
        └── __init__.py
```

---

## How to Run

### 1. Edit paths in `src/config.py`
```python
DATASET_ROOT = '/path/to/smart_cdl_mimo3x2'
SAVE_DIR     = '/path/to/checkpoints'
```

### 2. Train 5 ensemble members
```bash
# Sequential:
bash src/train/run_all_members.sh

# Parallel on Kaggle: open 5 notebooks, run each with a different seed:
python src/train/train_member.py --seed 0
python src/train/train_member.py --seed 1   # ... up to seed 4
```
Member diversity comes from per-seed RNG (weight init, shuffle order, AWGN draws).

### 3. Evaluate
```bash
python src/evaluate/ensemble_eval.py
```
Produces `ensemble_results.png` + `ensemble_results.npy` in `SAVE_DIR`.

---

## Dataset

Clean-physics pipeline: MATLAB generates raw CDL coefficients with **no noise**. `Y_clean` is
power-normalised in MATLAB (E[|Y|²]=1) and `sig_power` is saved so Python can restore the raw
physical scale. AWGN injection + a single shared physics-preserving normalisation happen in
Python at runtime.

**Corrected normalisation (`sample_to_real`):**
1. Un-normalise `Y` via `sqrt(sig_power)` so X, Y, H share one raw physical domain (fixes the "X raw + Y power-normed" scale mismatch).
2. Apply **one shared max-abs scale** to input and target — it cancels in NMSE and is recoverable at inference (multiply prediction by `scale` → physical H).
3. No mean subtraction (preserves LoS/DC phase).

This replaces the earlier per-sample target std (`h_sig`), which was not recoverable at deployment and inflated NMSE.

| Split | CDL Models | Purpose |
|---|---|---|
| train / val / test | A, C, D | Training, early stopping, in-distribution eval |
| gen_model (OOD) | B, E | Unseen channel families |
| gen_cond (OOD) | A, C, D (extreme DS/Doppler) | Unseen propagation conditions |

Kaggle dataset: `misaaew/smart-cdl-mimo3x2`

---

## Uncertainty Decomposition

```
μ*(x)      = mean_k μ_k(x)            ← ensemble prediction
aleatoric  = mean_k exp(logvar_k(x))  ← data noise (per NLL head)
epistemic  = var_k μ_k(x)            ← member disagreement (OOD detector)
σ_total    = sqrt(aleatoric + epistemic)
σ_cal      = w₁·√alea + w₂·√epis + w₃·spatial + w₀   ← post-hoc calibrated (fit on val)
```

Calibration weights (fit on val, corrected run): `w_alea=+1.47  w_epis=+1.16  w_spat=−0.026  bias=+0.0025`.
The large positive epistemic weight shows that after the normalisation fix, disagreement carries
real per-sample signal.

**Each component serves a different role — report them separately:**
- `aleatoric`: SNR-level calibration (per-SNR r = 0.98)
- `epistemic`: OOD detection (AUROC 0.99 pooled, 1.000 at SNR ≥ 15 dB, 66.4× elevation)
- `σ_cal`: within-SNR per-sample selective prediction (r = 0.45, retention curves)

---

## BNN Stability Constants

| Constant | Value | Reason |
|---|---|---|
| `INIT_SIGMA` | 0.05 | Start meaningfully stochastic; 1e-7 collapses immediately |
| `KL_WEIGHT` | 5e-8 | Fixed beta: target 3–10% KL contribution |
| `KL_WARMUP` | 50 epochs | Pure NLL first; avoids premature KL collapse |
| `clamp` | (−10, 4) | σ ∈ [exp(−5), exp(2)]; −6 floor removed |
| `alpha` | 0.1 | Gradient leak from encoder to logvar branch |
| `lv_out bias` | −2.0 | Init σ ≈ 0.37 |

---

## Positioning vs the literature

NMSE is competitive but not the headline — most baselines are SISO and report global NMSE in
the −13 to −18 dB range under harder evaluation; this is MIMO 3×2 with per-SNR reporting, so the
absolute dB are not a one-line comparison. The defensible, uncontested contribution is the
**uncertainty + OOD axis**, which no point estimator or generative-prior method provides. The
2026 JSAC diffusion-Bayesian work (BMCE) models a generative channel *prior*; this work quantifies
the estimator's predictive *posterior* uncertainty and detects OOD — a different and complementary
capability.

---

*IIT Delhi EE Dept · Supervisor: Prof. Rajoriya · Compute: Kaggle T4 × 2*