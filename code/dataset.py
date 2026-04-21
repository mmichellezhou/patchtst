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