"""
Dataset Loading and Preprocessing
-----------------------------------
Handles loading and preprocessing of the ETT (Electricity Transformer
Temperature) datasets for time series forecasting.

Contains:
- ETTDataset: PyTorch Dataset class for ETTh1, ETTh2, ETTm1, ETTm2
- Train/validation/test splitting
- Instance normalization and sliding window construction
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import config

# Standard ETT fixed splits used in Informer / PatchTST papers
_ETT_SPLITS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}


class ETTDataset(Dataset):
    def __init__(self, data_path, split="train", config=config):
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # load and drop date column if present
        df = pd.read_csv(data_path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        data = df.values.astype(np.float32)

        # split sizes: fixed for ETT (paper protocol), 70/10/20 ratios for everything else
        dataset_name = config.dataset_name
        if dataset_name in _ETT_SPLITS:
            n_train, n_val, n_test = _ETT_SPLITS[dataset_name]
            total = n_train + n_val + n_test     # ETT may have unused rows past this
        else:
            n = len(data)
            n_train = int(n * config.train_ratio)
            n_test  = int(n * config.test_ratio)
            n_val   = n - n_train - n_test
            total = n

        # Autoformer/PatchTST border convention: val and test are padded by seq_len at the
        # start so a sliding window can begin at the first true val/test prediction step.
        # Without this padding, ILI's tiny val (~97 rows) is shorter than seq_len=104 and
        # __len__ goes negative.
        borders = {
            "train": (0,                              n_train),
            "val":   (n_train  - self.seq_len,        n_train + n_val),
            "test":  (n_train + n_val - self.seq_len, total),
        }

        # global normalization using training set statistics (no data leakage)
        train_start, train_end = borders["train"]
        mean = data[train_start:train_end].mean(axis=0, keepdims=True)
        std  = data[train_start:train_end].std(axis=0,  keepdims=True) + 1e-8
        data = (data - mean) / std

        start, end = borders[split]
        self.data = data[start:end]

    def __len__(self):
        # number of sliding windows we can extract
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                                 # (seq_len, 7)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, 7)
        return torch.tensor(x), torch.tensor(y)


def get_dataloaders(config=config):
    train_dataset = ETTDataset(config.data_path, split="train", config=config)
    val_dataset   = ETTDataset(config.data_path, split="val",   config=config)
    test_dataset  = ETTDataset(config.data_path, split="test",  config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"x shape: {x.shape}")    # (batch_size, seq_len, 7)
    print(f"y shape: {y.shape}")    # (batch_size, pred_len, 7)