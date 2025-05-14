import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.serialization import add_safe_globals

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import sys
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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


class LitHybrid(LightningModule):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        lr: float,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = HybridLSTMAttn(
            input_dim=input_dim,
            seq_len=seq_len,
            forecast_horizon=horizon,
            lstm_hidden=128,
            lstm_layers=2,
            dense_hidden=128,
            dropout=0.1,
            num_heads=4,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        metrics = calculate_metrics(
            y.cpu().numpy(), y_hat.detach().cpu().numpy()
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", metrics["mae"], on_epoch=True)
        self.log("val_rmse", metrics["rmse"], on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["features"], batch["target"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        metrics = calculate_metrics(
            y.cpu().numpy(), y_hat.detach().cpu().numpy()
        )
        self.log("test_loss", loss)
        self.log("test_mae", metrics["mae"])
        self.log("test_rmse", metrics["rmse"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            ),
            "monitor": "val_loss"
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HybridLSTMAttn with Lightning")
    parser.add_argument("--dataset-dir", type=str, default="data/prepared/lstm_datasets")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    # standard logger
    logger = setup_logger(__name__, level=args.log_level)
    logger.info(f"Using Lightning with TensorBoardLogger")

    # load data
    ds_path = Path(settings.PROJECT_ROOT, args.dataset_dir)
    train_loader = torch.load(ds_path / "train_loader.pt", weights_only=False)
    val_loader = torch.load(ds_path / "val_loader.pt",   weights_only=False)
    test_loader = torch.load(ds_path / "test_loader.pt",  weights_only=False)

    sample = next(iter(train_loader))
    seq_len, input_dim = sample["features"].shape[1:]
    horizon = sample["target"].shape[1]
    logger.info(f"Detected input_dim={input_dim}, seq_len={seq_len}, horizon={horizon}")

    # logger and callbacks
    tb_logger = TensorBoardLogger(
        save_dir=Path(settings.PROJECT_ROOT, "tb_logs"), name="attention_lstm"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(settings.PROJECT_ROOT, "models/attention_lstm"),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min"
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[early_stop_cb, checkpoint_cb],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
    )

    # model
    lit_model = LitHybrid(
        input_dim=input_dim,
        seq_len=seq_len,
        horizon=horizon,
        lr=args.lr,
    )

    # train & test
    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader)

    logger.info("Training complete. Checkpoints and logs saved.")