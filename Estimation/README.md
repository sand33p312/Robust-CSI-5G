# Phase 1 — Channel Estimation with Bayesian Neural Networks

**Task**: Reconstruct full CSI `H` from LS pilot observations `Y` for 5G NR CDL channels.  
**Config**: MIMO 3×2 (3 Tx × 2 Rx) · 3GPP CDL-A/C/D train · CDL-B/E OOD · 3.5 GHz · 12 SC × 14 sym  
**Method**: Deep Ensemble (K=5) of Heteroscedastic BNN v4 members

---

## Results

| Metric | Value |
|---|---|
| NMSE @ 30 dB | **−40.28 dB** |
| NMSE @ 0 dB | −26.18 dB |
| Per-SNR Pearson r | **0.9985** |
| Within-SNR Pearson r (calibrated σ) | **0.42** (was −0.19 for single-model VI) |
| OOD epistemic elevation (CDL-B/E) | **69.9×** |
| SNR-matched AUROC vs CDL-B/E | **0.893** (0.974 @ 30 dB) |
| OOD in discarded 50% (mixed pool) | **97.8%** |
| Ensemble gain over best member | +0.64 dB free |

---

## Architecture — HeteroscedasticBNN v4 (Hybrid)

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
  │  BayesLinear 10752→256 │  Conv2d 64→32→12→12 (1×1)  │
  │  LayerNorm + ReLU   │  clamp(−10, 4)                 │
  │  BayesLinear 256→256│  → log_var [B, 2016]           │
  │  LayerNorm + ReLU   │                                │
  │  BayesLinear 256→2016│                               │
  │  → mean [B, 2016]   │                                │
  └─────────────────────┴────────────────────────────────┘
```

**Key design choices:**
- **MLP decoder for mean**: 3.3M params gives −40 dB. Pure CNN decoder hits −25.9 dB ceiling.
- **α=0.1 gradient leak**: lets aleatoric head see encoder gradients → within-SNR r 0.05→0.42
- **clamp(−10, 4)**: removes the −6 floor that was pinning σ² at 0.00248 above 10 dB
- **GroupNorm throughout**: BatchNorm breaks under MC weight sampling
- **Fixed KL weight**: adaptive beta creates positive feedback loops

---

## Code Structure

```
src/
├── config.py                    ← all hyperparameters and paths
├── dataset/
│   ├── cdl_dataset.py           ← SmartCDLDataset_Train/Eval + build_loaders()
│   └── __init__.py
├── model/
│   ├── bnn_layers.py            ← VIModule, BayesLinear, BayesConv2d, BayesResBlock, MultiScaleBranch
│   ├── heteroscedastic_bnn.py   ← HeteroscedasticBNN v4 + build_model()
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
# Sequential (CPU/single GPU):
bash src/train/run_all_members.sh

# Parallel on Kaggle: open 5 notebooks, run each with SEED=0..4 via:
python src/train/train_member.py --seed 0
python src/train/train_member.py --seed 1
# ... up to seed 4
```

### 3. Evaluate
```bash
python src/evaluate/ensemble_eval.py
```
Produces: `ensemble_results.png` + `ensemble_results.npy` in `SAVE_DIR`

---

## Dataset

Clean-physics pipeline: MATLAB generates raw CDL coefficients only (no noise, no normalisation).  
AWGN injection + z-score normalisation happen in Python at runtime.

| Split | CDL Models | Purpose |
|---|---|---|
| train | CDL-A, C, D | Training (1260 scenario combos) |
| val | CDL-A, C, D | Early stopping (fixed AWGN seeds) |
| test | CDL-A, C, D | In-distribution evaluation |
| gen_model | CDL-B, E | OOD: unseen channel families |
| gen_cond | CDL-A, C, D (extreme DS/Dop) | OOD: unseen propagation conditions |

Kaggle dataset: `misaaew/smart-cdl-mimo3x2`

---

## Uncertainty Decomposition

```
μ*(x)      = mean_k μ_k(x)           ← ensemble prediction
aleatoric  = mean_k exp(logvar_k(x))  ← data noise (per NLL head)
epistemic  = var_k μ_k(x)            ← member disagreement (OOD detector)
σ_total    = sqrt(aleatoric + epistemic)
σ_cal      = w₁√alea + w₂√epis + w₃·spatial + w₀  ← post-hoc calibrated
```

**Each component serves a different role — report them separately:**
- `aleatoric`: SNR-level calibration (per-SNR r = 0.999)
- `epistemic`: OOD detection (AUROC 0.97 @ 30 dB, 69.9× elevation)
- `σ_cal`: within-SNR per-sample selective prediction (r = 0.42)

---

## BNN Stability Constants

| Constant | Value | Reason |
|---|---|---|
| `INIT_SIGMA` | 0.05 | Start meaningfully stochastic; 1e-7 collapses immediately |
| `KL_WEIGHT` | 5e-8 | Fixed beta: target 5% KL contribution |
| `KL_WARMUP` | 50 epochs | Pure NLL first; avoids premature KL collapse |
| `clamp` | (−10, 4) | σ ∈ [exp(−5), exp(2)]; −6 floor removed |
| `alpha` | 0.1 | Gradient leak from encoder to logvar branch |

---

*IIT Delhi EE Dept · Supervisor: Prof. Rajoriya · Compute: Kaggle T4 × 2*