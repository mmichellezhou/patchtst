"""
Evaluation
-----------
Evaluates a trained PatchTST model on the test set and compares
results against the original paper's reported metrics (Table 3).

Contains:
- Test set inference
- MSE and MAE computation per dataset and prediction horizon
- Result export to results/tables/
"""

import os
import torch
from utils import compute_metrics


def evaluate(model, test_loader, device):
    """Run model on test_loader and return (MSE, MAE)."""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu()
            all_preds.append(pred)
            all_trues.append(y)
    preds = torch.cat(all_preds, dim=0)
    trues = torch.cat(all_trues, dim=0)
    return compute_metrics(preds, trues)


def load_and_evaluate(config, device=None):
    """Load the best checkpoint for config and evaluate on the test set."""
    from patchtst import PatchTST
    from dataset import get_dataloaders

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(config)
    model = PatchTST(config).to(device)

    ckpt_path = os.path.join(config.checkpoint_path, "best_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    mse, mae = evaluate(model, test_loader, device)
    return mse, mae
