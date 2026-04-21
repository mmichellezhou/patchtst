"""
PatchTST Model Architecture
----------------------------
Implements the PatchTST model from "A Time Series is Worth 64 Words:
Long-Term Forecasting with Transformers" (Nie et al., ICLR 2023).

Contains:
- Patching: splits univariate time series into subseries-level patches
- Transformer Encoder: maps patches to latent representations
- Prediction Head: linear layer to produce forecast output
"""