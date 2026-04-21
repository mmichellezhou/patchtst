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

from dataclasses import dataclass, field


@dataclass
class Config:
    # Dataset
    dataset_name: str = "ETTh1"
    target: str = "OT"
    features: str = "M"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # Patching
    patch_len: int = 16
    stride: int = 8
    seq_len: int = 336

    # Model
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    instance_norm: bool = True

    # Forecasting
    pred_len: int = 96

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    seed: int = 2021

    # Paths
    save_path: str = "../results"
    checkpoint_path: str = "../results/checkpoints"

    def __post_init__(self):
        # auto-derive data path from dataset name
        self.data_path = f"../data/ETDataset/ETT-small/{self.dataset_name}.csv"

        # smaller model for ETTh1/ETTh2 to avoid overfitting (per Appendix A.1.4)
        if self.dataset_name in ["ETTh1", "ETTh2"]:
            self.n_heads = 4
            self.d_model = 16
            self.d_ff = 128


config = Config()