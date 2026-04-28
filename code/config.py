"""
Configuration and Hyperparameters
-----------------------------------
Central location for all model and training hyperparameters.
Modify this file to run different experimental settings.

Contains:
- Model parameters (patch length, stride, look-back window, d_model, etc.)
- Training parameters (learning rate, batch size, epochs)
- Dataset and experiment settings
"""

import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../patchtst/code
_ROOT = os.path.dirname(_HERE)                        # .../patchtst


@dataclass
class Config:
    # dataset
    dataset_name: str = "ETTh1"
    n_channels: int = 7
    target: str = "OT"
    features: str = "M"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # patching
    patch_len: int = 16
    stride: int = 8
    seq_len: int = 336

    # model
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    instance_norm: bool = True

    # forecasting
    pred_len: int = 96

    # training
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    seed: int = 2021

    def __post_init__(self):
        # auto-derive data path from dataset name
        self.data_path = os.path.join(_ROOT, "data", "ETDataset", "ETT-small", f"{self.dataset_name}.csv")
        self.save_path = os.path.join(_ROOT, "results")
        self.checkpoint_path = os.path.join(_ROOT, "results", "checkpoints")

        # smaller model for ETTh1/ETTh2 to avoid overfitting (per Appendix A.1.4)
        if self.dataset_name in ["ETTh1", "ETTh2"]:
            self.n_heads = 4
            self.d_model = 16
            self.d_ff = 128


config = Config()