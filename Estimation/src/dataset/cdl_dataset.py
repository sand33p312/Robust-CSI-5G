"""
dataset/cdl_dataset.py
SmartCDL MIMO 3×2 dataset loaders.

Design principle (clean-physics pipeline):
  MATLAB generates raw channel coefficients only — NO noise, NO normalisation.
  Python injects AWGN + z-score normalisation at runtime for maximum flexibility.

Two dataset classes:
  SmartCDLDataset_Train   — random SNR + per-member phase augmentation each call
  SmartCDLDataset_Eval    — fixed noise (seeded), pre-computed into RAM for speed
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATASET_ROOT, BATCH_SIZE, SNR_TRAIN_DB, SNR_EVAL_DB,
    CNN_IN_CH, NSC, NSYM, OUTPUT_DIM,
)


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def load_h5_ri(h5path: str, key: str) -> np.ndarray:
    """Load MATLAB complex array stored as [real, imag] last axis.
    MATLAB is column-major → always apply .T on load."""
    with h5py.File(h5path, 'r') as f:
        arr = f[key][:].T          # MATLAB col-major → Python row-major
    return arr[..., 0] + 1j * arr[..., 1]


def load_h5_scalar(h5path: str, key: str) -> np.ndarray:
    """Load a 1-D scalar array (delay_spread, doppler, scs_kHz)."""
    with h5py.File(h5path, 'r') as f:
        arr = f[key][:]
    return arr.T.squeeze()


def verify_clean(h5path: str) -> None:
    """Assert that the HDF5 file was generated without noise (noise_in_matlab==0)."""
    with h5py.File(h5path, 'r') as f:
        flag = int(np.asarray(f.attrs.get('noise_in_matlab', -1)).item())
    assert flag == 0, f'Expected noise_in_matlab=0, got {flag} in {h5path}'


# ── Core preprocessing ─────────────────────────────────────────────────────────

def sample_to_real(Xc: np.ndarray, Yn: np.ndarray, Hc: np.ndarray):
    """
    Convert complex pilot/received/channel arrays → real-valued model inputs.

    Input layout:  Xc, Yn, Hc  each [antennas, SC, sym]  (complex)
    Returns:
      x      float32 [2*(NTX+NRX), NSC, NSYM]  normalised input
      h      float32 [OUTPUT_DIM]                normalised flattened channel
      h_sig  float32                             per-sample channel std (for denorm)
    """
    eps = 1e-8
    X_real = np.concatenate([Xc.real, Xc.imag], axis=0).astype(np.float32)
    Y_real = np.concatenate([Yn.real, Yn.imag], axis=0).astype(np.float32)
    H_real = np.concatenate([Hc.real, Hc.imag], axis=0).astype(np.float32)
    xy     = np.concatenate([X_real, Y_real], axis=0)
    xy_sig = xy.std()  + eps
    h_sig  = H_real.std() + eps
    return (xy - xy.mean()) / xy_sig, ((H_real - H_real.mean()) / h_sig).flatten(), h_sig


# ── Dataset classes ────────────────────────────────────────────────────────────

class SmartCDLDataset_Train(Dataset):
    """
    Training dataset — online augmentation per sample per epoch.

    Augmentation:
      - Random SNR drawn uniformly from SNR_TRAIN_DB each call
      - Physics-preserving global phase rotation e^{jφ} applied to X and Y
        (same rotation → H is unchanged in magnitude, phase offset cancels)
      - Member-unique noise trajectories: RNG seeded by (SEED+1)*100000 + idx
        so seeds 0..4 produce fully non-overlapping noise realisations

    SEED must be set before constructing the loader (see train/train_member.py).
    """

    def __init__(self, h5path: str, snr_db_list: list, seed: int = 0):
        verify_clean(h5path)
        self.seed        = seed
        self.X_c         = load_h5_ri(h5path, 'X_grid')
        self.Y_c         = load_h5_ri(h5path, 'Y_clean')
        self.H_c         = load_h5_ri(h5path, 'H_freq')
        self.delay_spread = load_h5_scalar(h5path, 'delay_spread')
        self.doppler      = load_h5_scalar(h5path, 'doppler_shift')
        self.scs_kHz      = load_h5_scalar(h5path, 'scs_kHz')
        self.snr_db_list  = snr_db_list
        with h5py.File(h5path, 'r') as f:
            self.N = int(np.asarray(f.attrs['num_samples']).item())
        assert self.X_c.shape[0] == self.N, "sample count mismatch"
        print(f'  [train] N={self.N}  random SNR {snr_db_list}  seed={seed}')

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        # 1. Member-unique deterministic RNG for augmentation
        rng_seed = int((self.seed + 1) * 100000 + idx)
        state    = np.random.RandomState(rng_seed)

        # 2. Random global phase rotation (physics-preserving: H unchanged)
        phase_shift = state.uniform(-0.04, 0.04)

        # 3. Member-unique AWGN noise
        snr    = float(np.random.choice(self.snr_db_list))
        np_val = 10 ** (-snr / 10)
        noise  = np.sqrt(np_val / 2) * (
            state.randn(*self.Y_c[idx].shape) +
            1j * state.randn(*self.Y_c[idx].shape)
        )

        # 4. Apply phase rotation to X and Y (not H — target stays clean)
        rot         = np.exp(1j * phase_shift)
        X_perturbed = self.X_c[idx] * rot
        Y_perturbed = (self.Y_c[idx] + noise) * rot

        # 5. Convert to real-valued normalised tensors
        x, h, h_sig = sample_to_real(X_perturbed, Y_perturbed, self.H_c[idx])

        return {
            'x':     torch.from_numpy(x),
            'h':     torch.from_numpy(h),
            'snr':   torch.tensor(snr,   dtype=torch.float32),
            'h_sig': torch.tensor(h_sig, dtype=torch.float32),
        }


class SmartCDLDataset_Eval(Dataset):
    """
    Evaluation dataset — fixed AWGN seeds, pre-computed into RAM.

    Each clean sample is replicated once per SNR level with a fixed seed
    (si*100000 + i) so validation NMSE is deterministic and early stopping
    is meaningful.  No phase augmentation.
    """

    def __init__(self, h5path: str, snr_db_list: list, split_name: str = 'eval'):
        verify_clean(h5path)
        print(f'  [{split_name}] Pre-applying noise...')
        X_c = load_h5_ri(h5path, 'X_grid')
        Y_c = load_h5_ri(h5path, 'Y_clean')
        H_c = load_h5_ri(h5path, 'H_freq')
        ds  = load_h5_scalar(h5path, 'delay_spread')
        dop = load_h5_scalar(h5path, 'doppler_shift')
        scs = load_h5_scalar(h5path, 'scs_kHz')
        with h5py.File(h5path, 'r') as f:
            N_clean = int(np.asarray(f.attrs['num_samples']).item())

        nSNR  = len(snr_db_list)
        N_tot = N_clean * nSNR
        xs    = np.zeros((N_tot, CNN_IN_CH, NSC, NSYM), dtype=np.float32)
        hs    = np.zeros((N_tot, OUTPUT_DIM),            dtype=np.float32)
        h_sigs = np.zeros(N_tot, dtype=np.float32)
        snrs   = np.zeros(N_tot, dtype=np.float32)
        dss    = np.zeros(N_tot, dtype=np.float32)
        dops   = np.zeros(N_tot, dtype=np.float32)
        scss   = np.zeros(N_tot, dtype=np.float32)

        for si, snr in enumerate(snr_db_list):
            np_val = 10 ** (-snr / 10)
            for i in range(N_clean):
                rng   = np.random.RandomState(seed=si * 100000 + i)
                noise = np.sqrt(np_val / 2) * (
                    rng.randn(*Y_c[i].shape) + 1j * rng.randn(*Y_c[i].shape)
                )
                x, h, h_sig = sample_to_real(X_c[i], Y_c[i] + noise, H_c[i])
                j = si * N_clean + i
                xs[j] = x; hs[j] = h; h_sigs[j] = h_sig
                snrs[j] = snr; dss[j] = ds[i]; dops[j] = dop[i]; scss[j] = scs[i]

        self.x           = torch.from_numpy(xs)
        self.h           = torch.from_numpy(hs)
        self.h_sig       = torch.from_numpy(h_sigs)
        self.snr         = torch.from_numpy(snrs)
        self.delay_spread = dss
        self.doppler      = dops
        self.scs_kHz      = scss
        self.N = N_tot; self.N_clean = N_clean; self.snr_db_list = snr_db_list
        # Keep raw arrays for potential downstream analysis
        self.X_c_raw = X_c; self.Y_c_raw = Y_c; self.H_c_raw = H_c
        print(f'  [{split_name}] {N_clean}×{nSNR}={N_tot} samples ✓')

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        return {
            'x':     self.x[idx],
            'h':     self.h[idx],
            'snr':   self.snr[idx],
            'h_sig': self.h_sig[idx],
        }


# ── Loader factory ─────────────────────────────────────────────────────────────

def h5p(split: str) -> str:
    return os.path.join(DATASET_ROOT, split, f'{split}.h5')


def build_loaders(seed: int = 0, include_gen: bool = True):
    """
    Build all DataLoaders.  seed is passed to the training dataset for
    member-unique augmentation.

    Returns: train_loader, val_loader, test_loader, gen_loaders (dict)
    """
    train_ds = SmartCDLDataset_Train(h5p('train'), SNR_TRAIN_DB, seed=seed)
    val_ds   = SmartCDLDataset_Eval(h5p('val'),   SNR_EVAL_DB,  'val')
    test_ds  = SmartCDLDataset_Eval(h5p('test'),  SNR_EVAL_DB,  'test')

    kw = dict(batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    gen_loaders = {}
    if include_gen:
        for split in ['gen_model', 'gen_cond']:
            p = h5p(split)
            if os.path.exists(p):
                ds = SmartCDLDataset_Eval(p, SNR_EVAL_DB, split)
                gen_loaders[split] = DataLoader(ds, shuffle=False, **kw)

    print(f'Ready: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}')
    return train_loader, val_loader, test_loader, gen_loaders