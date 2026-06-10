"""
config.py — Central hyperparameter and path configuration
BNN Channel Estimation | MIMO 3×2 | CDL | IIT Delhi
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = '/kaggle/input/datasets/misaaew/smart-cdl-mimo3x2/smart_cdl_mimo3x2'
SAVE_DIR     = '/kaggle/working/hetero_bnn_ensemble_v2'
CKPT_DIR     = SAVE_DIR   # where best_seed{k}.pt files are written / read

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Ensemble ───────────────────────────────────────────────────────────────────
N_MEMBERS = 5
CKPTS     = [f'{CKPT_DIR}/best_seed{k}.pt' for k in range(N_MEMBERS)]

# ── Training schedule ──────────────────────────────────────────────────────────
EPOCHS       = 500
BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 60
KL_WARMUP    = 50       # pure NLL for first 50 epochs before KL turns on
MC_TRAIN     = 3        # MC samples during training forward pass
MC_TEST      = 10       # MC samples during eval (not used for ensemble mean)

# ── Architecture ──────────────────────────────────────────────────────────────
CNN_CHANNELS = 64
HIDDEN_DIM   = 256
NUM_HIDDEN   = 2

# ── BNN stability constants (DO NOT CHANGE without full re-run) ───────────────
PRIOR_SIGMA = 1.0
INIT_SIGMA  = 0.05      # start meaningfully stochastic; 1e-7 collapses immediately
# KL_WEIGHT: v4 KL≈1.1M, |NLL|≈1.0 → target 5% → kl_w = 0.05/1.1e6 ≈ 4.5e-8
KL_WEIGHT   = 5e-8      # fixed beta (adaptive beta creates positive feedback loops)

# ── MIMO 3×2 dimensions ───────────────────────────────────────────────────────
NTX, NRX  = 3, 2
NCH       = NTX * NRX       # 6 channel pairs
NSC, NSYM = 12, 14
CNN_IN_CH = 2 * NTX + 2 * NRX  # 10 input channels (Re+Im for each antenna)
NCH_OUT   = NCH * 2            # 12 output channels (Re+Im for 6 pairs)
OUTPUT_DIM = NCH_OUT * NSC * NSYM  # 2016

# ── SNR grids ─────────────────────────────────────────────────────────────────
SNR_TRAIN_DB   = [-5, 0, 5, 10, 15, 20, 25, 30]
SNR_EVAL_DB    = [-5, 0, 5, 10, 15, 20, 25, 30]
EARLY_STOP_SNR = 20  # SNR level used for early stopping (val NMSE@20dB)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')