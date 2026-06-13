"""
train/train_member.py
Train one ensemble member.  Run 5 times with SEED = 0, 1, 2, 3, 4.

Usage:
    python train/train_member.py --seed 0
    python train/train_member.py --seed 1
    ...

Each run saves: SAVE_DIR/best_seed{SEED}.pt
(The evaluation script expects all 5 checkpoints in CKPT_DIR.)

Member diversity comes from the per-seed RNG: different weight initialisation,
different data-shuffle order, and different AWGN draws each epoch. (The earlier
per-member phase-augmentation was removed when the normalisation was corrected;
seed-driven diversity is sufficient for a deep ensemble.)

Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
  Preferred over ReduceLROnPlateau — restarts prevent local minima and
  give stable convergence without triggering on noisy val metrics.

KL schedule: fixed beta with linear warmup over KL_WARMUP epochs.
  Adaptive beta is NOT used — it creates positive feedback loops when
  KL drops and NLL rises, destabilising training.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim

# Allow running from repo root: python train/train_member.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from dataset.cdl_dataset import build_loaders
from model import build_model
from utils.metrics import elbo_loss, nmse_db_global


# ── Helpers ────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, device, kl_w):
    model.train()
    tn = tk = n = 0
    for batch in loader:
        x = batch['x'].to(device)
        h = batch['h'].to(device)
        opt.zero_grad()
        loss, nll, kl = elbo_loss(model, x, h, kl_w, cfg.MC_TRAIN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        bs = x.size(0); tn += nll * bs; tk += kl * bs; n += bs
    return tn / n, tk / n


@torch.no_grad()
def evaluate_per_snr(model, loader, device):
    model.eval()
    preds, trues, sigs, snrs = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        mean, _ = model(x, False)   # deterministic (mean weights)
        preds.append(mean.cpu()); trues.append(batch['h'])
        sigs.append(batch['scale']); snrs.append(batch['snr'])
    P = torch.cat(preds); T = torch.cat(trues)
    S = torch.cat(sigs);  N = torch.cat(snrs).numpy()
    return {int(s): nmse_db_global(P[N == s], T[N == s], S[N == s])
            for s in np.unique(N)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True,
                        help='Ensemble member index: 0..4')
    args = parser.parse_args()
    SEED = args.seed

    # Reproducibility
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    checkpoint = f'{cfg.SAVE_DIR}/best_seed{SEED}.pt'

    print(f'ENSEMBLE SEED : {SEED}  (run 5 times: 0,1,2,3,4)')
    print(f'Device        : {cfg.DEVICE}')
    print(f'OUTPUT_DIM    : {cfg.OUTPUT_DIM}  (= {cfg.NCH_OUT} ch × {cfg.NSC} SC × {cfg.NSYM} sym)')
    print(f'INIT_SIGMA    : {cfg.INIT_SIGMA}')
    print(f'KL_WEIGHT     : {cfg.KL_WEIGHT:.0e}')

    # Data
    train_loader, val_loader, _, _ = build_loaders(include_gen=False)

    # Model
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        _b = next(iter(train_loader))
        _m, _lv = model(_b['x'].to(cfg.DEVICE), False)
        assert _m.shape == (cfg.BATCH_SIZE, cfg.OUTPUT_DIM)
        _s = _lv.exp().sqrt().mean().item()
        print(f'  Shape check ✓  Init σ_mean={_s:.4f}  (target ~0.37)')
        assert abs(_s - 0.3679) < 0.05, f'σ_mean={_s:.4f} not ~0.37 — check lv_out bias init'
    del _b, _m, _lv, _s

    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=cfg.DEVICE))
        print('  Loaded checkpoint — resuming.')

    # Optimiser + scheduler
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-5)

    best = float('inf'); best_ep = 0; no_imp = 0

    print(f'\nTraining — Heteroscedastic BNN v4 | {cfg.DEVICE}')
    print(f'Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)')
    print(f'{"Ep":>4} | {"lr":>8} | {"kl_w":>8} | {"NLL":>9} | {"KL":>11} | {"KL%":>6} | {"Val@20":>8} | Best')

    for ep in range(1, cfg.EPOCHS + 1):
        # KL warmup: pure NLL for KL_WARMUP epochs, then linear ramp over same window
        kl_w = 0.0 if ep <= cfg.KL_WARMUP else (
            cfg.KL_WEIGHT * min(1.0, (ep - cfg.KL_WARMUP) / cfg.KL_WARMUP)
        )
        nll, kl = train_epoch(model, train_loader, opt, cfg.DEVICE, kl_w)
        val_snr = evaluate_per_snr(model, val_loader, cfg.DEVICE)
        vnmse   = val_snr[cfg.EARLY_STOP_SNR]
        sch.step(ep)

        if vnmse < best:
            best = vnmse; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            no_imp += 1

        if ep % 10 == 0 or ep == 1:
            kl_pct = (kl_w * kl) / (abs(nll) + 1e-12) * 100
            lr_now = opt.param_groups[0]['lr']
            flag = ('(warmup)' if ep <= cfg.KL_WARMUP
                    else '▲ high' if kl_pct > 10
                    else '▼ low'  if kl_pct < 1 else '✓')
            print(f'{ep:4d} | {lr_now:.1e} | {kl_w:.1e} | {nll:9.4f} | {kl:11.1f} | '
                  f'{kl_pct:5.1f}% | {vnmse:8.2f}dB | {best:.2f}@ep{best_ep} {flag}')

        if no_imp >= cfg.PATIENCE:
            print(f'Early stop @ ep {ep}'); break

    print(f'\nBest val NMSE @{cfg.EARLY_STOP_SNR}dB = {best:.2f} dB @ ep {best_ep}')
    print(f'Checkpoint: {checkpoint}')


if __name__ == '__main__':
    main()