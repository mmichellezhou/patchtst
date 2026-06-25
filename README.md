# PatchTST Reimplementation
<img width="1100" height="732" alt="Screenshot 2026-06-25 at 11 58 35 AM" src="https://github.com/user-attachments/assets/84705ce2-8ceb-4db3-9df0-c822a297a7a0" />


## 1. Introduction
This repo is a reimplementation of the PatchTST time-series forecasting model from "A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers" (Nie et al., ICLR 2023). The paper introduces PatchTST, a time series transformer that can outperform other transformer-based models through patching and channel independence.

## 2. Chosen Result
The target result is the PatchTST long-term forecasting performance on the ETT, illness, and weather datasets, matching the paper’s reported metrics in Table 3.
This reproduction evaluates whether the re-implementation can recover similar MSE/MAE scores for multi-horizon prediction.

## 3. GitHub Contents
- `code/`: implementation files for model, data loading, training, evaluation, and utils (everything is run in results.ipynb)
- `data/`: dataset download instructions and ETT CSV files
- `results/`: saved checkpoints, loss history, and predicted outputs
- `report/`: project report and notes

## 4. Re-implementation Details
This code implements PatchTST with patch-level tokenization, transformer encoder blocks, and optional cross-channel or multiscale patching variants.
It uses the ETT datasets (ETTh1/ETTh2/ETTm1/ETTm2/weather/illness), PyTorch training loops, MSE loss, and standard forecasting evaluation metrics.

## 5. Reproduction Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Download ETT data as described in `data/README.md`
3. Run cells in `results.ipynb`
A GPU is recommended for training, but CPU execution is possible for small-scale tests.
The repo produces PatchTST forecasts and stores best model checkpoints under `results/checkpoints/best_model.pt`.
Visualizations are stored under `results/figures`.

## 6. Results/Insights
The repo closely reproduces the paper's forecasting results with a 1-3% deviation in MSE and MAE. Using the PatchTSTCrossChannel and PatchTSTMultiScale models also produce similar results to the paper, though they are consistently slightly worse (this does depend on the patches chosen for PatchTSTMultiScale).

## 7. Conclusion
Our reimplementation reinforces the paper's central claim that patching and channel independence can create a more robust transformer model to predict time series.

## 8. References
- Nie, S., Wang, Y., Lin, H., Wang, Y., & Liu, C. (2023). "A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers." ICLR 2023.
- We used Claude to create the skeleton template for our codebase and repository.

## 9. Acknowledgements
This work was completed as a class final project for Cornell's CS4782: Deep Learning, with reference to the original PatchTST implementation.
