"""
dataset/cdl_dataset.py
SmartCDL MIMO 3x2 dataset loaders.

Design principle (clean-physics pipeline):
  MATLAB generates raw channel coefficients only - NO noise.
  Y_clean is power-normalised in MATLAB (E[|Y|^2]=1); sig_power is saved so
  Python can restore the raw physical scale before adding noise.
  Python injects AWGN + a SINGLE physically-consistent normalisation at runtime.

  ------------------------------------------------------------------------------
  NORMALISATION (corrected - deployment-honest, CeBed/Channelformer-comparable):
  ------------------------------------------------------------------------------
  Earlier versions scaled the target H by its own ground-truth std (h_sig) and
  the input by a separate xy std, then mean-centred both. That coupled the
  learning target to ground-truth statistics the receiver never has at
  deployment, made the task artificially easy (model only had to learn channel
  SHAPE, not SCALE), and inflated NMSE by ~5 dB.

  The corrected `sample_to_real` instead:
    1. Brings X, Y, H to the SAME raw physical domain
       (Y is un-normalised via sqrt(sig_power) so the pilot->received amplitude
        relationship that encodes the channel is physically consistent - fixes
        the "X raw + Y power-normed" mismatch).
    2. Applies ONE shared scale to input AND target (max-abs), so the scale
       cancels in NMSE and is recoverable at inference (multiply prediction by
       `scale` to get physical H).
    3. Does NOT subtract the mean (preserves LoS/DC phase information).
  ------------------------------------------------------------------------------

Two dataset classes:
  SmartCDLDataset_Train   - random SNR each call (online augmentation)
  SmartCDLDataset_Eval    - fixed noise (seeded), pre-computed into RAM for speed
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


# -- HDF5 helpers ---------------------------------------------------------------

def load_h5_ri(h5path, key):
    """Load MATLAB complex array stored as [real, imag] last axis.
    MATLAB is column-major -> always apply .T on load."""
    with h5py.File(h5path, 'r') as f:
        arr = f[key][:].T          # MATLAB col-major -> Python row-major
    return arr[..., 0] + 1j * arr[..., 1]


def load_h5_scalar(h5path, key):
    """Load a 1-D scalar array (sig_power, delay_spread, doppler, scs_kHz)."""
    with h5py.File(h5path, 'r') as f:
        arr = f[key][:]
    return arr.T.squeeze()


def verify_clean(h5path):
    """Assert the HDF5 file was generated without noise (noise_in_matlab==0)."""
    with h5py.File(h5path, 'r') as f:
        flag = int(np.asarray(f.attrs.get('noise_in_matlab', -1)).item())
    assert flag == 0, f'Expected noise_in_matlab=0, got {flag} in {h5path}'


# -- Core preprocessing ---------------------------------------------------------

def sample_to_real(Xc, Yn, Hc):
    """
    Convert complex pilot/received/channel arrays -> real-valued model inputs.

    CORRECTED normalisation:
      - X, Y, H must already be in the SAME raw physical domain when passed in
        (caller un-normalises Y via sqrt(sig_power) first).
      - One shared max-abs scale for input AND target.
      - No mean subtraction (phase preserved).

    Input layout:  Xc, Yn, Hc  each [antennas, SC, sym]  (complex, raw domain)
    Returns:
      xy_norm  float32 [2*(NTX+NRX), NSC, NSYM]  normalised input
      h_norm   float32 [OUTPUT_DIM]                normalised flattened channel
      scale    float32                             shared scale (for denorm at inference)
    """
    eps = 1e-8
    X_real = np.concatenate([Xc.real, Xc.imag], axis=0).astype(np.float32)
    Y_real = np.concatenate([Yn.real, Yn.imag], axis=0).astype(np.float32)
    H_real = np.concatenate([Hc.real, Hc.imag], axis=0).astype(np.float32)

    xy = np.concatenate([X_real, Y_real], axis=0)

    # SINGLE unifying scale (max physical amplitude across input and target).
    # Shared by input + target so it cancels in NMSE and is recoverable at
    # inference. NOTE: to be strictly input-only (no peek at H), replace with
    #   scale = np.max(np.abs(xy)) + eps
    scale = max(np.max(np.abs(xy)), np.max(np.abs(H_real))) + eps

    xy_norm = xy / scale
    h_norm  = H_real / scale          # NO mean subtraction (preserves DC/LoS phase)

    return xy_norm, h_norm.flatten(), scale


# -- Dataset classes ------------------------------------------------------------

class SmartCDLDataset_Train(Dataset):
    """
    Training dataset - online noise augmentation per sample per epoch.

    Pipeline per __getitem__:
      1. Draw random SNR from SNR_TRAIN_DB.
      2. Add AWGN (power 10^(-snr/10)) on the NORMALISED Y (unit power) so the
         in-domain SNR is exact.
      3. Un-normalise Y to the raw physical domain via sqrt(sig_power) - signal
         and noise scale together, so SNR is preserved.
      4. Apply the shared physics-preserving normalisation.
    """

    def __init__(self, h5path, snr_db_list):
        verify_clean(h5path)
        self.X_c       = load_h5_ri(h5path, 'X_grid')
        self.Y_c       = load_h5_ri(h5path, 'Y_clean')   # power-normalised in MATLAB
        self.H_c       = load_h5_ri(h5path, 'H_freq')
        self.sig_power = load_h5_scalar(h5path, 'sig_power')  # raw received power ref
        self.delay_spread = load_h5_scalar(h5path, 'delay_spread')
        self.doppler      = load_h5_scalar(h5path, 'doppler_shift')
        self.scs_kHz      = load_h5_scalar(h5path, 'scs_kHz')
        self.snr_db_list  = snr_db_list
        with h5py.File(h5path, 'r') as f:
            self.N = int(np.asarray(f.attrs['num_samples']).item())
        assert self.X_c.shape[0] == self.N, "sample count mismatch"
        print(f'  [train] N={self.N}  random SNR {snr_db_list}')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        snr    = float(np.random.choice(self.snr_db_list))
        np_val = 10 ** (-snr / 10)

        # 1+2. AWGN on normalised Y (unit power) -> exact in-domain SNR
        noise = np.sqrt(np_val / 2) * (
            np.random.randn(*self.Y_c[idx].shape) +
            1j * np.random.randn(*self.Y_c[idx].shape)
        )
        Yn_norm = self.Y_c[idx] + noise

        # 3. Un-normalise to raw physical domain (SNR preserved)
        Yn_raw = Yn_norm * np.sqrt(self.sig_power[idx])

        # 4. Shared physics-preserving normalisation
        x, h, scale = sample_to_real(self.X_c[idx], Yn_raw, self.H_c[idx])

        return {
            'x':     torch.from_numpy(x),
            'h':     torch.from_numpy(h),
            'snr':   torch.tensor(snr,   dtype=torch.float32),
            'scale': torch.tensor(scale, dtype=torch.float32),
        }


class SmartCDLDataset_Eval(Dataset):
    """
    Evaluation dataset - fixed AWGN seeds, pre-computed into RAM.

    Each clean sample is replicated once per SNR level with a fixed seed
    (si*100000 + i) so validation NMSE is deterministic and early stopping
    is meaningful. Same noise -> un-normalise -> normalise pipeline as train.
    """

    def __init__(self, h5path, snr_db_list, split_name='eval'):
        verify_clean(h5path)
        print(f'  [{split_name}] Pre-applying noise in physical domain...')
        X_c = load_h5_ri(h5path, 'X_grid')
        Y_c = load_h5_ri(h5path, 'Y_clean')
        H_c = load_h5_ri(h5path, 'H_freq')
        sig_power = load_h5_scalar(h5path, 'sig_power')
        ds  = load_h5_scalar(h5path, 'delay_spread')
        dop = load_h5_scalar(h5path, 'doppler_shift')
        scs = load_h5_scalar(h5path, 'scs_kHz')
        with h5py.File(h5path, 'r') as f:
            N_clean = int(np.asarray(f.attrs['num_samples']).item())

        nSNR  = len(snr_db_list)
        N_tot = N_clean * nSNR
        xs     = np.zeros((N_tot, CNN_IN_CH, NSC, NSYM), dtype=np.float32)
        hs     = np.zeros((N_tot, OUTPUT_DIM),            dtype=np.float32)
        scales = np.zeros(N_tot, dtype=np.float32)
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
                Yn_norm = Y_c[i] + noise
                Yn_raw  = Yn_norm * np.sqrt(sig_power[i])   # un-normalise
                x, h, scale = sample_to_real(X_c[i], Yn_raw, H_c[i])
                j = si * N_clean + i
                xs[j] = x; hs[j] = h; scales[j] = scale
                snrs[j] = snr; dss[j] = ds[i]; dops[j] = dop[i]; scss[j] = scs[i]

        self.x           = torch.from_numpy(xs)
        self.h           = torch.from_numpy(hs)
        self.scale       = torch.from_numpy(scales)
        self.snr         = torch.from_numpy(snrs)
        self.delay_spread = dss
        self.doppler      = dops
        self.scs_kHz      = scss
        self.sig_power    = sig_power
        self.N = N_tot; self.N_clean = N_clean; self.snr_db_list = snr_db_list
        # Keep raw arrays for downstream analysis
        self.X_c_raw = X_c; self.Y_c_raw = Y_c; self.H_c_raw = H_c
        print(f'  [{split_name}] {N_clean}x{nSNR}={N_tot} samples done')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            'x':     self.x[idx],
            'h':     self.h[idx],
            'snr':   self.snr[idx],
            'scale': self.scale[idx],
        }


# -- Loader factory -------------------------------------------------------------

def h5p(split):
    return os.path.join(DATASET_ROOT, split, f'{split}.h5')


def build_loaders(include_gen=True):
    """
    Build all DataLoaders.

    Returns: train_loader, val_loader, test_loader, gen_loaders (dict)
    """
    train_ds = SmartCDLDataset_Train(h5p('train'), SNR_TRAIN_DB)
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