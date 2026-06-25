# Reproducing PatchTST

<img width="1204" height="802" alt="patchtst" src="https://github.com/user-attachments/assets/49434ada-32bb-4b02-bcef-44b7d4f5ee99" />

A reimplementation and evaluation of **PatchTST**, introduced in *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers* by Nie et al. (ICLR 2023).

## Overview

Time-series forecasting is important across energy, weather, healthcare, and financial applications. Recurrent models such as RNNs and LSTMs can struggle with long sequences, while standard Transformers apply attention to every timestep, increasing computational cost.

PatchTST addresses these challenges by dividing each time-series channel into patches and processing each channel independently with shared Transformer weights. This project reproduces the paper’s supervised multivariate long-term forecasting results and evaluates whether the implementation can achieve similar MSE and MAE scores across several benchmark datasets.

## Chosen Result

We focused on reproducing the **PatchTST/42** long-term forecasting results reported in Table 3 of the original paper. These experiments evaluate forecasting performance across electricity, illness, and weather datasets using multiple prediction horizons.

This result supports the paper’s central claim that patch-based tokenization and channel independence can outperform earlier Transformer-based forecasting models.
 
 <img width="154" height="151" alt="image" src="https://github.com/user-attachments/assets/2e6d06b4-ea9b-476e-a526-0d4aa1802706" />

*Figure 1. Table 3 results from the original paper.*


## Methodology

### Model Architecture

Each input sequence uses a lookback window of 336 timesteps and processes every channel independently. Each channel is divided into overlapping patches with a patch length of 16 and a stride of 8. These patches are treated as individual tokens and projected into a 128-dimensional embedding space with positional embeddings.

Reversible Instance Normalization is applied independently to each sample and channel. The Transformer encoder includes:

* 3 encoder layers
* 128-dimensional embeddings
* 16 attention heads
* 256-dimensional feed-forward layers
* Residual connections
* Batch normalization
* Dropout of 0.2

Transformer weights are shared across all channels. The output patch embeddings are flattened and passed through a linear prediction head that maps the representations to the selected forecasting horizon.

### Training and Evaluation

The model was evaluated on the following datasets:

* ETTh1
* ETTh2
* ETTm1
* ETTm2
* Illness
* Weather

Prediction horizons include 96, 192, 336, and 720 timesteps. The ETT datasets use a 12-month training split, a 4-month validation split, and a 4-month testing split.

Training uses Adam with a learning rate of `1e-4`, a batch size of 128, and a maximum of 100 epochs. Early stopping is based on validation MSE.

Performance is measured using:

* Mean Squared Error
* Mean Absolute Error

## Extensions

We implemented two extensions to the original architecture.

### Cross-Channel Attention

Patches from every channel are combined into a single token sequence using positional and channel embeddings. This allows attention to operate across variables before the outputs are reshaped back into individual channels.

### Multi-Scale Patching

The input is processed in parallel using patch sizes of 8, 16, and 32. The resulting representations are combined before generating the final forecast, allowing the model to capture patterns across multiple temporal scales.

## Results

Our implementation reproduced the original paper’s forecasting results with MSE and MAE values generally within **1–3%** of the reported metrics. Small differences may result from random seeds and minor variations in hyperparameters or implementation details.

The reproduction also preserved the qualitative patterns reported in the paper, including the greater difficulty of forecasting the ETTh datasets compared with the ETTm datasets.
<img width="270" height="158" alt="image" src="https://github.com/user-attachments/assets/f3e51813-d84f-4d32-9d47-f17ac48087e1" />


*Figure 2. MSE and MAE comparison between our implementation and the original paper.*
<img width="179" height="162" alt="image" src="https://github.com/user-attachments/assets/bc9691d8-999c-4971-a333-0f7aad03cab5" />


*Figure 3. MSE comparison between multi-scale patching and standard PatchTST.*
<img width="175" height="153" alt="image" src="https://github.com/user-attachments/assets/8cad7273-e474-4713-b266-686e63c06ef2" />


*Figure 4. MSE comparison between cross-channel attention and standard PatchTST.*
<img width="396" height="162" alt="image" src="https://github.com/user-attachments/assets/04d9dd6b-dcbd-4079-98ba-a7d53fad1555" />


*Figure 5. Forecast predictions across the six evaluated datasets.*

Multi-scale patching captured finer temporal changes but performed slightly worse overall. Cross-channel attention also underperformed the standard model, suggesting that combining channels may introduce more noise and spurious correlations than useful information.

These results reinforce the paper’s conclusion that channel independence provides a strong inductive bias for long-term time-series forecasting.

## Reflections

Implementation details such as RevIN and dropout had a major effect on model performance. Omitting these components during early experiments produced noticeably worse results.

Channel independence was also more effective than expected. Although cross-channel attention appeared intuitively useful, it often introduced noise instead of meaningful information. Patching improved more than computational efficiency—it also helped the model learn broader temporal patterns rather than relying on isolated timesteps.

Future work could explore better combinations of patch sizes or selective cross-channel attention applied only within certain Transformer layers.

## Reproduction

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the datasets using the instructions in `data/README.md`.

Run the cells in:

```text
results.ipynb
```

A GPU is recommended for full training runs, although smaller experiments can be run on a CPU.

Saved outputs include:

```text
results/checkpoints/best_model.pt
results/figures/
```

## Repository Structure

```text
code/       Model, data loading, training, evaluation, and utility files
data/       Dataset files and download instructions
results/    Checkpoints, predictions, loss history, and figures
report/     Project report and supporting notes
```

## References

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers.* International Conference on Learning Representations.

