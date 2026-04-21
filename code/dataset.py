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


class ETTDataset(Dataset):
    def __init__(self, data_path, split="train", config=config):
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.instance_norm = config.instance_norm

        # load and drop date column
        df = pd.read_csv(data_path)
        df = df.drop(columns=["date"])
        data = df.values.astype(np.float32)  # (17420, 7)

        # train/val/test split
        n = len(data)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        if split == "train":
            self.data = data[:train_end]
        elif split == "val":
            self.data = data[train_end:val_end]
        elif split == "test":
            self.data = data[val_end:]

    def __len__(self):
        # number of sliding windows we can extract
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]           # (seq_len, 7)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, 7)

        # instance normalization per channel
        if self.instance_norm:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mean) / std
            y = (y - mean) / std  # use same mean/std from input window

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
    print(f"x shape: {x.shape}")  # (batch_size, seq_len, 7)
    print(f"y shape: {y.shape}")  # (batch_size, pred_len, 7)