"""
Utility Functions
------------------
Shared helper functions used across training and evaluation.

Contains:
- MSE and MAE metric computation
- Plot generation (forecast vs. actual, loss curves)
- Result saving to results/figures/ and results/tables/
<<<<<<< HEAD
"""
=======
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_metrics(pred: torch.Tensor, true: torch.Tensor):
    """Return (MSE, MAE) as plain Python floats."""
    mse = ((pred - true) ** 2).mean().item()
    mae = (pred - true).abs().mean().item()
    return mse, mae


def plot_loss_curves(train_losses, val_losses, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train MSE")
    ax.plot(val_losses,   label="val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_forecast(x_true, y_true, y_pred, channel=0, title="Forecast vs. Actual", save_path=None):
    """Plot one channel of a single sample: input context + ground truth + prediction."""
    seq_len  = x_true.shape[0]
    pred_len = y_true.shape[0]
    t_in  = np.arange(seq_len)
    t_out = np.arange(seq_len, seq_len + pred_len)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_in,  x_true[:, channel],  color="steelblue",  label="input")
    ax.plot(t_out, y_true[:, channel],  color="steelblue",  linestyle="--", label="ground truth")
    ax.plot(t_out, y_pred[:, channel],  color="tomato",     label="prediction")
    ax.axvline(seq_len, color="gray", linestyle=":")
    ax.set_xlabel("Time step")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
>>>>>>> origin/main
