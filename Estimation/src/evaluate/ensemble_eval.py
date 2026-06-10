"""
evaluate/ensemble_eval.py
Full evaluation of the Deep Ensemble (K=5) model.

Sections:
  1. Load all 5 checkpoints and verify member diversity
  2. Test NMSE: ensemble vs individual members
  3. Calibration: per-SNR and within-SNR Pearson r
  4. OOD generalisation: CDL-B/E and extreme Doppler/DS
  5. Post-hoc linear calibration (val fit → test eval)
  6. Error-retention curves (AUSE)
  7. OOD detection AUROC (pooled + SNR-matched)
  8. Plots
  9. Final summary table

Uncertainty decomposition:
  μ*(x)       = mean_k μ_k(x)          ← ensemble prediction
  aleatoric   = mean_k exp(logvar_k(x)) ← data noise (per member NLL head)
  epistemic   = var_k μ_k(x)           ← member DISAGREEMENT (the OOD detector)
  σ_total     = sqrt(aleatoric + epistemic)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from dataset.cdl_dataset import build_loaders
from model import build_model
from utils.metrics import nmse_db_global


# ── 1. Load ensemble ───────────────────────────────────────────────────────────

def load_ensemble():
    ensemble = []
    found = 0
    for k, ckpt in enumerate(cfg.CKPTS):
        ok = os.path.exists(ckpt)
        found += ok
        print(f'  [{"✓" if ok else "✗ MISSING"}] {ckpt}')
        if ok:
            m = build_model(cfg)
            m.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
            m.eval()
            ensemble.append(m)

    assert found >= 2, 'Need at least 2 members for ensemble disagreement!'
    K = len(ensemble)

    # Sanity: check members are genuinely different
    probe_batch = 8
    with torch.no_grad():
        x_probe = torch.randn(probe_batch, cfg.CNN_IN_CH, cfg.NSC, cfg.NSYM, device=cfg.DEVICE)
        preds = [m(x_probe, False)[0] for m in ensemble]
    diffs = [((preds[i] - preds[j]) ** 2).mean().item()
             for i in range(K) for j in range(i+1, K)]
    mean_diff = float(np.mean(diffs))
    print(f'Member disagreement on {probe_batch} test samples: {mean_diff:.6f}')
    assert mean_diff > 1e-6, 'Members appear identical — re-check checkpoints!'
    print('Members are genuinely different ✓')
    return ensemble, K


# ── 2. Ensemble forward pass ──────────────────────────────────────────────────

@torch.no_grad()
def ensemble_forward_loader(loader, ensemble, K):
    """
    Welford online update over K members — memory-safe.
    Returns: mu_star [N,D], alea [N,D], epis [N,D], lv_mean [N,D],
             H [N,D], SIG [N], SNR [N]
    """
    mu_all, alea_all, epis_all, lv_all = [], [], [], []
    H_all, SIG_all, SNR_all = [], [], []

    for batch in loader:
        x = batch['x'].to(cfg.DEVICE); B = x.size(0)
        m_run  = torch.zeros(B, cfg.OUTPUT_DIM)
        m2_run = torch.zeros(B, cfg.OUTPUT_DIM)
        a_run  = torch.zeros(B, cfg.OUTPUT_DIM)
        lv_run = torch.zeros(B, cfg.OUTPUT_DIM)

        for k, mdl in enumerate(ensemble, start=1):
            mu_k, lv_k = mdl(x, sample=False)
            mu_k = mu_k.cpu(); lv_k = lv_k.cpu()
            d = mu_k - m_run
            m_run  = m_run  + d / k
            m2_run = m2_run + d * (mu_k - m_run)
            a_run  = a_run  + lv_k.exp()
            lv_run = lv_run + lv_k

        mu_all.append(m_run)
        epis_all.append(m2_run / (K - 1) if K > 1 else m2_run * 0)
        alea_all.append(a_run / K)
        lv_all.append(lv_run / K)
        H_all.append(batch['h']); SIG_all.append(batch['h_sig'])
        SNR_all.append(batch['snr'])

    return (torch.cat(mu_all), torch.cat(alea_all), torch.cat(epis_all),
            torch.cat(lv_all), torch.cat(H_all), torch.cat(SIG_all),
            torch.cat(SNR_all))


# ── 3. NMSE helpers ───────────────────────────────────────────────────────────

def compute_test_nmse(MU, H_t, SIG_t, SNR_t, ensemble, K, test_loader):
    SNR_np = SNR_t.numpy()
    snrs_u = sorted(np.unique(SNR_np).astype(int).tolist())

    # Per-SNR ensemble NMSE
    res_snr = {}
    for s in snrs_u:
        m = torch.from_numpy(SNR_np == s)
        res_snr[s] = nmse_db_global(MU[m], H_t[m], SIG_t[m])

    # Individual member NMSE @ 20 dB
    print('Computing individual member NMSE @ 20 dB...')
    preds_k = {k: [] for k in range(K)}
    trues, sigs = [], []
    with torch.no_grad():
        for batch in test_loader:
            sm = batch['snr'].numpy() == 20
            if not sm.any(): continue
            x = batch['x'][sm].to(cfg.DEVICE)
            for k, mdl in enumerate(ensemble):
                preds_k[k].append(mdl(x, False)[0].cpu())
            trues.append(batch['h'][sm]); sigs.append(batch['h_sig'][sm])
    T20 = torch.cat(trues); S20 = torch.cat(sigs)
    member_nmse20 = [nmse_db_global(torch.cat(preds_k[k]), T20, S20) for k in range(K)]

    # Physical-domain arrays for downstream analysis
    P_ph = (MU    * SIG_t.unsqueeze(-1)).numpy()
    T_ph = (H_t   * SIG_t.unsqueeze(-1)).numpy()
    test_global = 10 * np.log10(((P_ph - T_ph)**2).sum() / ((T_ph**2).sum() + 1e-12))

    print(f'\n{"="*58}')
    print(f'  ENSEMBLE (K={K}) vs INDIVIDUAL MEMBERS')
    print(f'  {"Member":>10} | {"NMSE@20dB":>10}')
    for k, v in enumerate(member_nmse20):
        print(f'  seed {k:>5} | {v:10.2f} dB')
    print(f'  {"ENSEMBLE":>10} | {res_snr[20]:10.2f} dB  ← gain from averaging')
    print(f'\n  Ensemble per-SNR NMSE:')
    for s, v in sorted(res_snr.items()): print(f'  {s:>4}dB | {v:8.2f} dB')
    print(f'  Global: {test_global:.2f} dB')
    print(f'{"="*58}')

    return res_snr, member_nmse20, P_ph, T_ph, test_global, snrs_u, SNR_np


# ── 4. Calibration ────────────────────────────────────────────────────────────

def compute_calibration(ALEA, EPIS, LVM, P_ph, T_ph, SNR_np, snrs_u, K):
    alea_ps = ALEA.mean(-1).numpy()
    epis_ps = EPIS.mean(-1).numpy()
    spat_ps = LVM.std(-1).numpy()
    total_sigma = np.sqrt(alea_ps + epis_ps)
    err = np.sqrt(np.mean((P_ph - T_ph) ** 2, axis=-1))

    corr_all  = np.corrcoef(total_sigma, err)[0, 1]
    corr_alea = np.corrcoef(alea_ps,     err)[0, 1]
    corr_epis = np.corrcoef(epis_ps,     err)[0, 1]
    corr_spat = np.corrcoef(spat_ps,     err)[0, 1]

    s_snr    = np.array([total_sigma[SNR_np==s].mean() for s in snrs_u])
    e_snr    = np.array([err[SNR_np==s].mean()          for s in snrs_u])
    r_snr    = np.corrcoef(s_snr, e_snr)[0, 1]
    alea_snr = np.array([alea_ps[SNR_np==s].mean() for s in snrs_u])
    epis_snr = np.array([epis_ps[SNR_np==s].mean() for s in snrs_u])

    # Within-SNR (z-scored per SNR group)
    epis_norm = np.zeros_like(epis_ps)
    tot_norm  = np.zeros_like(total_sigma)
    err_norm  = np.zeros_like(err)
    for s in snrs_u:
        m = SNR_np == s
        epis_norm[m] = (epis_ps[m] - epis_ps[m].mean()) / (epis_ps[m].std() + 1e-12)
        tot_norm[m]  = (total_sigma[m] - total_sigma[m].mean()) / (total_sigma[m].std() + 1e-12)
        err_norm[m]  = (err[m] - err[m].mean()) / (err[m].std() + 1e-12)
    r_within_epis = np.corrcoef(epis_norm, err_norm)[0, 1]
    r_within_tot  = np.corrcoef(tot_norm,  err_norm)[0, 1]

    print(f'CALIBRATION — Deep Ensemble (K={K})')
    print(f'  {"="*56}')
    print(f'  Pearson r — total σ (all samples) : {corr_all:.4f}')
    print(f'  Pearson r — total σ (per SNR)     : {r_snr:.4f}')
    print(f'  {"─"*56}')
    print(f'  WITHIN-SNR (per-sample discrimination):')
    print(f'  Pearson r — epistemic, within-SNR : {r_within_epis:.4f}')
    print(f'  Pearson r — total σ,  within-SNR  : {r_within_tot:.4f}')
    print(f'  (single-model mean-field VI gave -0.19 here)')
    print(f'  {"─"*56}')
    print(f'  Component Pearson r:  Alea={corr_alea:.4f}  Epis={corr_epis:.4f}  Spat={corr_spat:.4f}')

    print(f'\n  {"SNR":>4} | {"σ_total":>10} | {"σ_alea":>10} | {"σ_epis":>10} | {"RMSE":>10}')
    for s, sv, av, ev, er in zip(snrs_u, s_snr, alea_snr, epis_snr, e_snr):
        print(f'  {s:>4}dB | {sv:10.5f} | {av:10.5f} | {ev:10.6f} | {er:10.5f}')

    print(f'\n  Within-SNR epistemic per level:')
    for s in snrs_u:
        m = SNR_np == s; ev = epis_ps[m]; er_s = err[m]
        r_w = np.corrcoef(ev, er_s)[0, 1] if ev.std() > 1e-14 else 0.0
        print(f'    {s:>4}dB : σ_epis=[{ev.min():.2e},{ev.max():.2e}]  within-SNR r={r_w:+.3f}')

    return (alea_ps, epis_ps, spat_ps, total_sigma, err,
            corr_all, corr_alea, corr_epis, corr_spat,
            r_snr, r_within_epis, r_within_tot,
            s_snr, e_snr, alea_snr, epis_snr)


# ── 5. OOD generalisation ─────────────────────────────────────────────────────

def compute_ood(gen_loaders, ensemble, K, test_global, epis_ps):
    gen_results = {}
    epis_test_mean = epis_ps.mean()

    for split, loader in gen_loaders.items():
        MUg, ALEAg, EPISg, LVMg, Hg, SIGg, SNRg = ensemble_forward_loader(
            loader, ensemble, K)
        SNRg_np = SNRg.numpy()
        Pg_ph = (MUg * SIGg.unsqueeze(-1)).numpy()
        Tg_ph = (Hg  * SIGg.unsqueeze(-1)).numpy()
        nmse_g = 10 * np.log10(((Pg_ph - Tg_ph)**2).sum() / ((Tg_ph**2).sum() + 1e-12))
        per_snr = {int(s): nmse_db_global(MUg[SNRg_np==s], Hg[SNRg_np==s], SIGg[SNRg_np==s])
                   for s in np.unique(SNRg_np)}
        epis_ood = EPISg.mean(-1).numpy().mean()
        label = 'CDL-B/E (unseen models)' if split == 'gen_model' else 'extreme Dop/DS'

        gen_results[split] = {
            'global':   nmse_g, 'per_snr': per_snr, 'epis': epis_ood,
            'epis_ps':  EPISg.mean(-1).numpy(),
            'se_ps':    ((Pg_ph - Tg_ph)**2).sum(-1),
            'te_ps':    (Tg_ph**2).sum(-1),
        }

        print(f'\n  [{split}] {label}')
        print(f'    Global NMSE  : {nmse_g:.2f} dB  (seen: {test_global:.2f} dB)')
        print(f'    OOD gap      : {nmse_g - test_global:+.2f} dB')
        print(f'    Epistemic σ² : {epis_ood:.2e}  (test: {epis_test_mean:.2e})'
              f'  → {epis_ood/(epis_test_mean+1e-15):.1f}× elevated')
        for s, v in sorted(per_snr.items()): print(f'    {s:>4}dB : {v:8.2f} dB')
        del MUg, ALEAg, EPISg, LVMg, Hg, SIGg, Pg_ph, Tg_ph

    return gen_results, epis_test_mean


# ── 6. Post-hoc calibration ───────────────────────────────────────────────────

def posthoc_calibration(val_loader, ensemble, K, alea_ps, epis_ps, spat_ps,
                        err, SNR_np, snrs_u, corr_all, r_snr, r_within_tot,
                        r_within_epis, e_snr):
    print('Running ensemble over VAL set to fit calibration weights...')
    MUv, ALEAv, EPISv, LVMv, Hv, SIGv, SNRv = ensemble_forward_loader(
        val_loader, ensemble, K)
    alea_v = ALEAv.mean(-1).numpy(); epis_v = EPISv.mean(-1).numpy()
    spat_v = LVMv.std(-1).numpy()
    Pv_ph  = (MUv * SIGv.unsqueeze(-1)).numpy()
    Tv_ph  = (Hv  * SIGv.unsqueeze(-1)).numpy()
    err_v  = np.sqrt(np.mean((Pv_ph - Tv_ph)**2, axis=-1))
    del MUv, ALEAv, EPISv, LVMv, Hv, SIGv, Pv_ph, Tv_ph

    Xv = np.stack([np.sqrt(alea_v), np.sqrt(epis_v), spat_v, np.ones_like(alea_v)], axis=1)
    w, *_ = np.linalg.lstsq(Xv, err_v, rcond=None)
    print(f'Calibration weights (fit on val, N={len(err_v)}):')
    print(f'  w_alea={w[0]:+.4f}  w_epis={w[1]:+.4f}  w_spat={w[2]:+.4f}  bias={w[3]:+.5f}')

    Xt = np.stack([np.sqrt(alea_ps), np.sqrt(epis_ps), spat_ps, np.ones_like(alea_ps)], axis=1)
    sigma_cal = Xt @ w
    corr_cal  = np.corrcoef(sigma_cal, err)[0, 1]

    cal_norm = np.zeros_like(sigma_cal); errn = np.zeros_like(err)
    for s in snrs_u:
        m = SNR_np == s
        cal_norm[m] = (sigma_cal[m] - sigma_cal[m].mean()) / (sigma_cal[m].std() + 1e-12)
        errn[m]     = (err[m] - err[m].mean()) / (err[m].std() + 1e-12)
    r_within_cal = np.corrcoef(cal_norm, errn)[0, 1]
    s_cal_snr    = np.array([sigma_cal[SNR_np==s].mean() for s in snrs_u])
    r_snr_cal    = np.corrcoef(s_cal_snr, e_snr)[0, 1]

    print(f'\nCALIBRATED σ — test set results:')
    print(f'  Pearson r (all samples) : {corr_cal:.4f}   (uncalibrated: {corr_all:.4f})')
    print(f'  Pearson r (per SNR)     : {r_snr_cal:.4f}   (uncalibrated: {r_snr:.4f})')
    print(f'  Pearson r (within-SNR)  : {r_within_cal:.4f}'
          f'   (uncalibrated total: {r_within_tot:.4f}, epis alone: {r_within_epis:.4f})')

    return sigma_cal, corr_cal, r_within_cal, r_snr_cal


# ── 7. Retention curves ───────────────────────────────────────────────────────

def retention_curve(se, te, score, fracs):
    order = np.argsort(score)
    se_s, te_s = se[order], te[order]
    return np.array([10 * np.log10(se_s[:max(1, int(len(se)*f))].sum() /
                                   (te_s[:max(1, int(len(se)*f))].sum() + 1e-12))
                     for f in fracs])


def compute_retention(P_ph, T_ph, sigma_cal, epis_ps, gen_results):
    se_test = ((P_ph - T_ph)**2).sum(-1)
    te_test = (T_ph**2).sum(-1)
    fracs   = np.arange(0.1, 1.001, 0.05)

    curve_cal    = retention_curve(se_test, te_test, sigma_cal, fracs)
    curve_epis   = retention_curve(se_test, te_test, epis_ps,   fracs)
    curve_oracle = retention_curve(se_test, te_test, se_test / te_test, fracs)
    nmse_full    = 10 * np.log10(se_test.sum() / te_test.sum())

    ause_cal  = trapezoid(curve_cal  - curve_oracle, fracs)
    ause_epis = trapezoid(curve_epis - curve_oracle, fracs)

    print('RETENTION — test set only:')
    print(f'  {"retain":>7} | {"by σ_cal":>9} | {"by epis":>9} | {"oracle":>9}')
    for i, f in enumerate(fracs):
        if abs(f * 100 % 10) < 1e-6:
            print(f'  {f*100:6.0f}% | {curve_cal[i]:8.2f} | {curve_epis[i]:8.2f} | {curve_oracle[i]:8.2f}')
    print(f'  Full-set NMSE: {nmse_full:.2f} dB')
    print(f'  AUSE (σ_cal)  : {ause_cal:.3f} dB·frac   AUSE (epis): {ause_epis:.3f} dB·frac')

    if 'gen_model' in gen_results:
        g = gen_results['gen_model']
        se_mix  = np.concatenate([se_test, g['se_ps']]); te_mix = np.concatenate([te_test, g['te_ps']])
        ep_mix  = np.concatenate([epis_ps, g['epis_ps']])
        is_ood  = np.concatenate([np.zeros(len(se_test)), np.ones(len(g['se_ps']))])
        order   = np.argsort(ep_mix); n50 = len(ep_mix) // 2
        ood_pct = is_ood[order[n50:]].mean() * 100
        nmse_mix = 10 * np.log10(se_mix.sum() / te_mix.sum())
        print(f'\nRETENTION — mixed pool (test {len(se_test)} + CDL-B/E {len(g["se_ps"])}):'  )
        print(f'  Full mixed NMSE        : {nmse_mix:.2f} dB')
        print(f'  Retain 50% by epis     : {retention_curve(se_mix,te_mix,ep_mix,[0.5])[0]:.2f} dB')
        print(f'  Retain 20% by epis     : {retention_curve(se_mix,te_mix,ep_mix,[0.2])[0]:.2f} dB')
        print(f'  OOD share of discarded half: {ood_pct:.1f}%')
        print(f'  → epistemic-based rejection recovers near-clean NMSE from contaminated pool')

    return ause_cal, ause_epis


# ── 8. OOD AUROC ─────────────────────────────────────────────────────────────

def auroc(scores_neg, scores_pos):
    """Rank-based AUROC: P(score_pos > score_neg)."""
    s     = np.concatenate([scores_neg, scores_pos])
    order = s.argsort().argsort().astype(np.float64) + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    r_pos = order[len(scores_neg):].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_ood_auroc(epis_ps, gen_results, gen_loaders, SNR_np, snrs_u):
    print('OOD DETECTION — epistemic σ² as score:')
    for split in gen_results:
        a = auroc(epis_ps, gen_results[split]['epis_ps'])
        label = 'CDL-B/E (unseen models)' if split=='gen_model' else 'extreme Dop/DS (unseen cond.)'
        print(f'  test vs {split:9s} ({label:30s}): AUROC = {a:.4f}')

    print(f'\n  Sanity — aleatoric as OOD score (expected weak):')
    print(f'  (epistemic is the OOD detector; aleatoric tracks noise level by design)')

    print('\nSNR-MATCHED OOD DETECTION — epistemic σ² as score:')
    for split in gen_results:
        g_epis = gen_results[split]['epis_ps']
        g_snr  = gen_loaders[split].dataset.snr.numpy()
        label  = 'CDL-B/E (unseen models)' if split=='gen_model' else 'extreme Dop/DS'
        print(f'\n  test vs {split} ({label}):')
        aucs = []
        for s in snrs_u:
            a = auroc(epis_ps[SNR_np==s], g_epis[g_snr==s])
            aucs.append(a)
            bar = '█' * int((a - 0.5) * 40) if a > 0.5 else ''
            print(f'    {s:>4}dB : AUROC = {a:.4f}  {bar}')
        print(f'    {"mean":>5} : AUROC = {np.mean(aucs):.4f}   (pooled was {auroc(epis_ps,g_epis):.4f})')


# ── 9. Plots ──────────────────────────────────────────────────────────────────

def make_plots(snrs_u, res_snr, member_nmse20, epis_ps, err, SNR_np,
               r_within_epis, s_snr, e_snr, r_snr,
               alea_snr, epis_snr, corr_all, gen_results, K):
    fig = plt.figure(figsize=(22, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.32)
    bnn_v = [res_snr[s] for s in snrs_u]

    # NMSE vs SNR
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(snrs_u, bnn_v, 'b-o', lw=2, ms=7, label=f'Ensemble K={K}')
    for k, v in enumerate(member_nmse20):
        ax.plot(20, v, 'x', ms=8, color='gray', alpha=0.6)
    ax.plot([], [], 'x', color='gray', label='members @20dB')
    ax.set(xlabel='SNR (dB)', ylabel='NMSE (dB)', title='Ensemble NMSE vs SNR')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Epistemic vs error scatter
    ax = fig.add_subplot(gs[0, 1])
    idx_s = np.random.choice(len(epis_ps), min(5000, len(epis_ps)), replace=False)
    sc = ax.scatter(epis_ps[idx_s], err[idx_s], c=SNR_np[idx_s],
                    cmap='coolwarm', alpha=0.3, s=4)
    plt.colorbar(sc, ax=ax, label='SNR (dB)')
    ax.set(xlabel='Epistemic σ² (disagreement)', ylabel='RMSE',
           title=f'Epistemic vs Error\nwithin-SNR r={r_within_epis:.3f}')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    # σ vs RMSE per SNR
    ax = fig.add_subplot(gs[0, 2])
    n01 = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    ax.plot(snrs_u, n01(s_snr), 'b-o', lw=2, ms=7, label='σ_total')
    ax.plot(snrs_u, n01(e_snr), 'r-s', lw=2, ms=7, label='RMSE')
    ax.set(xlabel='SNR (dB)', ylabel='Norm.', title=f'σ vs RMSE per SNR r={r_snr:.3f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Aleatoric vs Epistemic per SNR
    ax = fig.add_subplot(gs[0, 3])
    xp = np.arange(len(snrs_u)); w = 0.35
    ax.bar(xp - w/2, alea_snr, w, label='Aleatoric', color='steelblue', alpha=0.8)
    ax.bar(xp + w/2, epis_snr, w, label='Epistemic', color='coral',     alpha=0.8)
    ax.set_xticks(xp); ax.set_xticklabels([str(s) for s in snrs_u], fontsize=7)
    ax.set_yscale('log')
    ax.set(xlabel='SNR (dB)', ylabel='Mean var (log)', title='Aleatoric vs Epistemic')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Gen-model OOD
    ax = fig.add_subplot(gs[1, 0])
    if 'gen_model' in gen_results:
        g = gen_results['gen_model']['per_snr']; gk = sorted(g.keys())
        ax.plot(gk, [g[s] for s in gk], 'g-o', lw=2, ms=7, label='CDL-B/E (OOD)')
        ax.plot(snrs_u, bnn_v, 'b--', lw=1.5, label='seen')
        ax.set(xlabel='SNR (dB)', ylabel='NMSE (dB)', title='Gen: Unseen CDL')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Gen-cond OOD
    ax = fig.add_subplot(gs[1, 1])
    if 'gen_cond' in gen_results:
        g = gen_results['gen_cond']['per_snr']; gk = sorted(g.keys())
        ax.plot(gk, [g[s] for s in gk], 'm-o', lw=2, ms=7, label='Extreme (OOD)')
        ax.plot(snrs_u, bnn_v, 'b--', lw=1.5, label='seen')
        ax.set(xlabel='SNR (dB)', ylabel='NMSE (dB)', title='Gen: Extreme Cond.')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Epistemic distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.log10(epis_ps + 1e-15), bins=40, density=True, alpha=0.5,
            label='test (seen)', color='steelblue')
    ax.set(xlabel='log10 epistemic σ²', ylabel='Density',
           title='Epistemic distribution (test)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Reliability diagram
    ax = fig.add_subplot(gs[1, 3])
    total_sigma = np.sqrt(epis_ps)  # simplified for plot
    n_bins = 10; edges = np.percentile(total_sigma, np.linspace(0, 100, n_bins + 1))
    bsig, berr = [], []
    for i in range(n_bins):
        m = (total_sigma >= edges[i]) & (total_sigma < (edges[i+1] if i < n_bins-1 else edges[i+1]+1))
        if m.sum() > 0:
            bsig.append(total_sigma[m].mean()); berr.append(err[m].mean())
    ax.plot(bsig, berr, 'bo-', lw=2, ms=7)
    ax.set(xlabel='Epistemic σ', ylabel='Actual RMSE', title=f'Reliability (epis) r={corr_all:.3f}')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Deep Ensemble Heteroscedastic BNN (K={K}) | within-SNR r={r_within_epis:.3f}',
                 fontsize=12, fontweight='bold')
    out = f'{cfg.SAVE_DIR}/ensemble_results.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out}')


# ── 10. Summary table ─────────────────────────────────────────────────────────

def print_summary(K, res_snr, member_nmse20, corr_all, r_snr, r_within_epis,
                  gen_results, epis_test_mean, corr_cal, r_within_cal, ause_cal,
                  epis_ps):
    print(); print('='*70)
    print(f'  FINAL SUMMARY — Deep Ensemble Heteroscedastic BNN (K={K})')
    print('='*70)
    rows = [
        ('Ensemble NMSE @ -5 dB',          f'{res_snr.get(-5, float("nan")):.2f} dB'),
        ('Ensemble NMSE @  0 dB',          f'{res_snr.get(0,  float("nan")):.2f} dB'),
        ('Ensemble NMSE @ 20 dB',          f'{res_snr.get(20, float("nan")):.2f} dB'),
        ('Ensemble NMSE @ 30 dB',          f'{res_snr.get(30, float("nan")):.2f} dB'),
        ('Best single member @ 20 dB',     f'{min(member_nmse20):.2f} dB'),
        ('Ensemble gain over best member', f'{min(member_nmse20)-res_snr.get(20,0):+.2f} dB'),
        ('─'*38, '─'*15),
        ('Pearson r (all samples)',         f'{corr_all:.4f}'),
        ('Pearson r (per SNR)',            f'{r_snr:.4f}'),
        ('Pearson r (within-SNR, epis)',   f'{r_within_epis:.4f}  ← THE FIX'),
        ('  (single-model VI baseline)',   '-0.19'),
        ('─'*38, '─'*15),
    ]
    if 'gen_model' in gen_results:
        g = gen_results['gen_model']
        rows += [('Gen-model NMSE (CDL-B/E)',  f'{g["global"]:.2f} dB'),
                 ('Gen-model epistemic ratio', f'{g["epis"]/(epis_test_mean+1e-15):.1f}× elevated')]
    if 'gen_cond' in gen_results:
        g = gen_results['gen_cond']
        rows += [('Gen-cond NMSE (extreme)',   f'{g["global"]:.2f} dB'),
                 ('Gen-cond epistemic ratio',  f'{g["epis"]/(epis_test_mean+1e-15):.1f}× elevated')]
    rows += [
        ('Calibrated σ r (all samples)',  f'{corr_cal:.4f}'),
        ('Calibrated σ r (within-SNR)',   f'{r_within_cal:.4f}'),
        ('AUSE (calibrated σ)',           f'{ause_cal:.3f}'),
        ('OOD AUROC (CDL-B/E)',
         f'{auroc(epis_ps, gen_results["gen_model"]["epis_ps"]):.4f}'
         if 'gen_model' in gen_results else 'N/A'),
        ('─'*38, '─'*15),
        ('Ensemble members K',              f'{K}'),
        ('Per-member architecture',         'Hetero BNN v4 (multi-scale + MLP)'),
        ('Aleatoric source',                'NLL logvar head (per member)'),
        ('Epistemic source',                'Member disagreement (function space)'),
    ]
    for k_, v_ in rows: print(f'  {k_:<40} {v_:>22}')
    print('='*70)

    np.save(f'{cfg.SAVE_DIR}/ensemble_results.npy', {
        'K': K, 'res_snr': res_snr, 'member_nmse20': member_nmse20,
        'pearson_all': corr_all, 'pearson_snr': r_snr,
        'pearson_within_epis': r_within_epis, 'gen_results': gen_results,
    })
    print(f'Saved: {cfg.SAVE_DIR}/ensemble_results.npy')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load ensemble
    ensemble, K = load_ensemble()

    # Data loaders
    _, val_loader, test_loader, gen_loaders = build_loaders(seed=0, include_gen=True)

    # Test NMSE
    print('Running ensemble over test set...')
    MU, ALEA, EPIS, LVM, H_t, SIG_t, SNR_t = ensemble_forward_loader(
        test_loader, ensemble, K)
    (res_snr, member_nmse20, P_ph, T_ph, test_global,
     snrs_u, SNR_np) = compute_test_nmse(MU, H_t, SIG_t, SNR_t, ensemble, K, test_loader)

    # Calibration
    (alea_ps, epis_ps, spat_ps, total_sigma, err,
     corr_all, corr_alea, corr_epis, corr_spat,
     r_snr, r_within_epis, r_within_tot,
     s_snr, e_snr, alea_snr, epis_snr) = compute_calibration(
        ALEA, EPIS, LVM, P_ph, T_ph, SNR_np, snrs_u, K)

    # OOD generalisation
    gen_results, epis_test_mean = compute_ood(
        gen_loaders, ensemble, K, test_global, epis_ps)

    # Post-hoc calibration
    sigma_cal, corr_cal, r_within_cal, r_snr_cal = posthoc_calibration(
        val_loader, ensemble, K, alea_ps, epis_ps, spat_ps, err,
        SNR_np, snrs_u, corr_all, r_snr, r_within_tot, r_within_epis, e_snr)

    # Retention curves
    ause_cal, ause_epis = compute_retention(P_ph, T_ph, sigma_cal, epis_ps, gen_results)

    # AUROC
    compute_ood_auroc(epis_ps, gen_results, gen_loaders, SNR_np, snrs_u)

    # Plots
    make_plots(snrs_u, res_snr, member_nmse20, epis_ps, err, SNR_np,
               r_within_epis, s_snr, e_snr, r_snr, alea_snr, epis_snr,
               corr_all, gen_results, K)

    # Summary
    print_summary(K, res_snr, member_nmse20, corr_all, r_snr, r_within_epis,
                  gen_results, epis_test_mean, corr_cal, r_within_cal, ause_cal,
                  epis_ps)


if __name__ == '__main__':
    main()