

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def compute_lyapunov_multivariate(initial_sequence, model, steps=200, delta=1e-5):
    """
    Estimate largest Lyapunov exponent (LLE) for multivariate sequence using a trained model.
    Uses Wolf-style renormalization to keep perturbation stable.

    Args:
        initial_sequence: shape (seq_len, num_features), ground truth seed sequence
        model: trained PyTorch model
        steps: number of forward steps
        delta: small perturbation size

    Returns:
        lyap: estimated LLE
        t_vals: array of time steps
        divergences: log divergence values
    """
    seq_len, num_features = initial_sequence.shape
    device = next(model.parameters()).device

    # Base trajectory
    x0 = initial_sequence.reshape(1, seq_len, num_features).astype(np.float32)

    x1 = np.copy(x0)
    x1[0, -1, 0] += delta

    d0 = delta  
    divergences = []

    for t in range(steps):
        with torch.no_grad():
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)

            y0 = model(x0_tensor).cpu().numpy()
            y1 = model(x1_tensor).cpu().numpy()

        diff = y1 - y0
        d = np.linalg.norm(diff)

        if d > 1e-12:
            divergences.append(np.log(d / d0))

            direction = diff / d
            y1 = y0 + d0 * direction
        else:
            y1 = y0 + d0 * np.random.randn(*y0.shape)

        x0 = np.roll(x0, -1, axis=1)
        x0[0, -1, :] = y0
        x1 = np.roll(x1, -1, axis=1)
        x1[0, -1, :] = y1

    divergences = np.array(divergences)
    t_vals = np.arange(len(divergences))

    coeffs = np.polyfit(t_vals, divergences, 1)
    lyap = coeffs[0]

    return lyap, t_vals, divergences


df = pd.read_csv("lorenz28.csv")  # your multivariate dataset
features = df[['y', 'z']].values   # <-- use exactly 2 features (same as training)
init_seq = features[:60]           # <-- use 60 timesteps (same as training seq_length)

lyap, t_vals, div_series = compute_lyapunov_multivariate(init_seq, model, steps=200)

print("Estimated Largest Lyapunov Exponent:", lyap)

plt.figure(figsize=(8,5))
plt.plot(t_vals, div_series, label="log divergence")
plt.plot(t_vals, np.poly1d(np.polyfit(t_vals, div_series, 1))(t_vals),
         'r--', label=f"Fit slope = {lyap:.4f}")
plt.xlabel("Time steps")
plt.ylabel("log(d/d0)")
plt.title("Lyapunov Exponent Estimation (Multivariate, Renormalized)")
plt.legend()
plt.show()
