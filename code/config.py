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

# Per-dataset hyperparameters from the official PatchTST/42 scripts
# (yuqinie98/PatchTST: PatchTST_supervised/scripts/PatchTST/*.sh).
# Fields override the Config defaults below.
_DATASET_SPECS = {
    "ETTh1":       {"n_channels": 7,   "d_model": 16,  "n_heads": 4,  "d_ff": 128, "dropout": 0.3, "batch_size": 128, "learning_rate": 1e-4,  "seq_len": 336},
    "ETTh2":       {"n_channels": 7,   "d_model": 16,  "n_heads": 4,  "d_ff": 128, "dropout": 0.3, "batch_size": 128, "learning_rate": 1e-4,  "seq_len": 336},
    "ETTm1":       {"n_channels": 7,   "d_model": 128, "n_heads": 16, "d_ff": 256, "dropout": 0.2, "batch_size": 128, "learning_rate": 1e-4,  "seq_len": 336},
    "ETTm2":       {"n_channels": 7,   "d_model": 128, "n_heads": 16, "d_ff": 256, "dropout": 0.2, "batch_size": 128, "learning_rate": 1e-4,  "seq_len": 336},
    "weather":     {"n_channels": 21,  "d_model": 128, "n_heads": 16, "d_ff": 256, "dropout": 0.2, "batch_size": 128, "learning_rate": 1e-4,  "seq_len": 336},
    "electricity": {"n_channels": 321, "d_model": 128, "n_heads": 16, "d_ff": 256, "dropout": 0.2, "batch_size": 16,  "learning_rate": 1e-4,  "seq_len": 336},
    "traffic":     {"n_channels": 862, "d_model": 128, "n_heads": 16, "d_ff": 256, "dropout": 0.2, "batch_size": 8,   "learning_rate": 1e-4,  "seq_len": 336},
    # ILI uses different patch_len/stride and a much shorter look-back
    "ili":         {"n_channels": 7,   "d_model": 16,  "n_heads": 4,  "d_ff": 128, "dropout": 0.3, "batch_size": 16,  "learning_rate": 2.5e-3, "seq_len": 104, "patch_len": 24, "stride": 2},
}


def _data_path_for(dataset_name: str) -> str:
    if dataset_name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        return os.path.join(_ROOT, "data", "ETDataset", "ETT-small", f"{dataset_name}.csv")
    if dataset_name.lower() == "ili":
        return os.path.join(_ROOT, "data", "national_illness.csv")
    # weather, traffic, electricity all live as <name>.csv
    return os.path.join(_ROOT, "data", f"{dataset_name.lower()}.csv")


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
    cross_channel_attention: bool = False
    use_multiscale_patches: bool = False
    patch_scales: tuple[int, ...] = field(default_factory=lambda: (8, 16, 32))

    # forecasting
    pred_len: int = 96

    # training
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    seed: int = 2021

    # paths
    save_path: str = "../results"
    checkpoint_path: str = "../results/checkpoints"

    def __post_init__(self):
        # apply per-dataset overrides from PatchTST/42 official scripts,
        # but only for fields the user did not explicitly override at construction time.
        spec = _DATASET_SPECS.get(self.dataset_name) or _DATASET_SPECS.get(self.dataset_name.lower())
        if spec is not None:
            defaults = Config.__dataclass_fields__
            for k, v in spec.items():
                # Only overwrite if the current value still matches the dataclass default
                # (i.e. the user did not pass it explicitly).
                if getattr(self, k) == defaults[k].default:
                    setattr(self, k, v)

        self.data_path = _data_path_for(self.dataset_name)
        self.save_path = os.path.join(_ROOT, "results")
        self.checkpoint_path = os.path.join(_ROOT, "results", "checkpoints")


config = Config()
