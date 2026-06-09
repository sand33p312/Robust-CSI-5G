# 📡 5G NR Smart MIMO 3x2 Channel Estimation Dataset

This dataset provides a physics-compliant 5G NR simulated environment (~200,000 samples) to train and stress-test Deep Learning models under real-world and Out-Of-Distribution (OOD) conditions.

## 🧠 "Clean Physics" Design
* **No Noise & No Normalization in MATLAB:** Data is saved as raw, complex physics.
* **Dynamic Pipeline:** Inject target SNRs (AWGN) and apply normalizations dynamically during the Python/PyTorch training loop.

## 📂 Generalization Splits
| Split | CDL Models | Condition | Purpose |
| :--- | :--- | :--- | :--- |
| **`train/`** | A, C, D | Standard | 70% pool for model training. |
| **`val/`** | A, C, D | Standard | 20% pool for validation/tuning. |
| **`test/`** | A, C, D | Standard | 10% pool for baseline testing. |
| **`gen_model/`** | B, E | Standard | **OOD Model:** Tests unseen structural geometries. |
| **`gen_cond/`** | A, C, D | Extreme | **OOD Condition:** Extreme Doppler, Delay Spread, and SCS. |

## 📐 Data Format (HDF5)
Data is stored as `.h5` files optimized for PyTorch dataloading. Complex arrays are `float32` with a final dimension of size `2` `(Real, Imaginary)`.

* **`/X_grid`**: `[N, 3, nSC, nSym, 2]` — Sparse pilot grid (Input).
* **`/Y_clean`**: `[N, 2, nSC, nSym, 2]` — Clean received grid, power-normalized to $E[|Y|^2]=1.0$.
* **`/H_freq`**: `[N, 6, nSC, nSym, 2]` — Ground-truth channel matrix for 3x2 links (Target).
* **Metadata Labels:** `/delay_spread`, `/doppler_shift`, `/scs_kHz`, `/H_power_dB`, `/sig_power`.

## 🛠️ Usage: Python Noise Injection
Because `Y_clean` is already power-normalized, adding specific SNR levels during PyTorch training is simple. Always apply noise *before* standardizing data.

```python
import torch
import numpy as np

def add_dynamic_noise(y_clean, snr_db):
    # Calculate required noise power
    noise_power = 10 ** (-snr_db / 10)
    
    # Generate complex AWGN 
    noise_real = torch.randn_like(y_clean[..., 0]) * np.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(y_clean[..., 1]) * np.sqrt(noise_power / 2)
    
    # Stack and add to clean signal
    noise = torch.stack([noise_real, noise_imag], dim=-1)
    return y_clean + noise

