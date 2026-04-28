"""
Training Loop
--------------
Trains the PatchTST model on a specified ETT dataset.

Contains:
- Training and validation loop with MSE loss
- Checkpoint saving
- Logging of train/val loss per epoch
"""

import os
import torch
import torch.nn as nn
from config import config
from dataset import get_dataloaders
from patchtst import PatchTST


def train(config=config):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_path, exist_ok=True)
    os.makedirs(config.save_path, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(config)
    model = PatchTST(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    patience, patience_counter = 20, 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(1, config.epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validate
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch:3d}/{config.epochs}  |  train: {avg_train_loss:.4f}  |  val: {avg_val_loss:.4f}")

        # checkpoint and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.checkpoint_path, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    torch.save({"train": train_losses, "val": val_losses},
               os.path.join(config.save_path, "loss_history.pt"))
    print(f"\nDone. Best val MSE: {best_val_loss:.4f}")
    return train_losses, val_losses


if __name__ == "__main__":
    train(config)
