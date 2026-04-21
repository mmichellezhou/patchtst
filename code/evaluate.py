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