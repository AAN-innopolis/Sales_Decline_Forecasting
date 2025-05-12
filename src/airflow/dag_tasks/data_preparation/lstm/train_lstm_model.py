import argparse
import logging
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.serialization import add_safe_globals

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.config.configs import settings
from src.utils.data_utils import setup_logger
from src.models.attention_lstm import HybridLSTMAttn  

add_safe_globals([
    "torch.utils.data.dataloader.DataLoader",
    "torch._utils._rebuild_tensor_v2",
    "torch.storage._TypedStorage",
    "collections.OrderedDict",
])

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE and RMSE (per-sample averaged over horizon)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train or eval for one epoch; return loss and (preds, targets)."""
    phase = "train" if train else "val"
    model.train(mode=train)
    epoch_loss = 0.0
    preds, targs = [], []

    bar = tqdm(loader, desc=f"{phase:>5}", leave=False)
    for batch in bar:
        x = batch["features"].to(device) # (B, T, F)
        assert not torch.isnan(x).any(), "NaNs found in input features!"
        assert not torch.isinf(x).any(), "Infs found in input features!"
        y = batch["target"].to(device) # (B, H)

        with torch.set_grad_enabled(train):
            y_hat = model(x) # (B, H)
            loss = criterion(y_hat, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        epoch_loss += loss.item() * x.size(0)
        preds.append(y_hat.detach().cpu().numpy())
        targs.append(y.detach().cpu().numpy())

        bar.set_postfix(loss=f"{loss.item():.4f}")

    preds  = np.concatenate(preds,  axis=0)
    targs  = np.concatenate(targs,  axis=0)
    epoch_loss /= len(loader.dataset)
    return epoch_loss, preds, targs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HybridLSTMAttn model")
    parser.add_argument("--dataset-dir", type=str,
                        default="data/prepared/lstm_datasets",
                        help="Folder with train_loader.pt etc.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logger = setup_logger(__name__, level=args.log_level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ds_path = Path(settings.PROJECT_ROOT, args.dataset_dir)
    logger.info(f"Loading DataLoaders from {ds_path}")

    train_loader = torch.load(ds_path / "train_loader.pt", weights_only=False)
    val_loader = torch.load(ds_path / "val_loader.pt",   weights_only=False)
    test_loader = torch.load(ds_path / "test_loader.pt",  weights_only=False)

    if args.batch_size != train_loader.batch_size:
        train_loader.batch_size = args.batch_size
        val_loader.batch_size = args.batch_size
        test_loader.batch_size = args.batch_size

    sample = next(iter(train_loader))
    seq_len, input_dim = sample["features"].shape[1:]
    horizon = sample["target"].shape[1]
    logger.info(f"Detected input_dim={input_dim}, seq_len={seq_len}, horizon={horizon}")

    model = HybridLSTMAttn(
        input_dim=input_dim,
        seq_len=seq_len,
        forecast_horizon=horizon,
        lstm_hidden=128,
        lstm_layers=2,
        dense_hidden=128,
        dropout=0.1,
        num_heads=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # training loop with early stopping
    best_val = float("inf")
    patience_left = args.patience
    history = {"train": [], "val": [], "mae": [], "rmse": []}

    ckpt_dir = Path(settings.PROJECT_ROOT, "models/attention_lstm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device,
            train=True, optimizer=optimizer
        )
        val_loss, v_pred, v_true = run_epoch(
            model, val_loader, criterion, device, train=False
        )
        metrics = calculate_metrics(v_true, v_pred)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["mae"].append(metrics["mae"])
        history["rmse"].append(metrics["rmse"])

        logger.info(
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"ValMAE={metrics['mae']:.4f} | "
            f"ValRMSE={metrics['rmse']:.4f}"
        )

        # early stopping & checkpoint
        if val_loss < best_val:
            best_val = val_loss
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss": best_val,
                },
                ckpt_dir / "best_model.pth",
            )
            logger.info("New best model saved.")
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pth")["model_state"])
    test_loss, t_pred, t_true = run_epoch(model, test_loader, criterion, device, False)
    test_metrics = calculate_metrics(t_true, t_pred)
    logger.info(
        f"\n*** Test results ***  "
        f"Loss={test_loss:.4f}  MAE={test_metrics['mae']:.4f}  "
        f"RMSE={test_metrics['rmse']:.4f}"
    )

    fig_dir = ckpt_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Loss curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "loss_curves.png")

    # forecast vs truth (first sample of val set)
    days = np.arange(1, horizon + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(days, v_true[0], label="actual")
    plt.plot(days, v_pred[0], label="pred")
    plt.xlabel("day"); plt.ylabel("target")
    plt.title("30‑day forecast (sample)"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "forecast_vs_truth.png")
    logger.info(f"Charts saved in {fig_dir}")


if __name__ == "__main__":
    main()