"""
Script for training the Temporal Fusion Transformer (TFT) model.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import torch.optim.lr_scheduler as lr_scheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.data_utils import setup_logger
from src.config.configs import settings


def load_datasets(
    data_dir: Path, 
    logger: logging.Logger
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet | None]:
    """
    Load training, validation, and optionally test datasets.

    Args:
        data_dir: Directory where datasets are stored.
        logger: Logger instance.

    Returns:
        A tuple containing training, validation, and test datasets.
        Test dataset can be None if not found.
    """
    logger.info(f"Loading datasets from {data_dir}...")
    try:
        training_dataset = torch.load(data_dir / "training_dataset.pt", weights_only=False)
        logger.info("Training dataset loaded.")
    except FileNotFoundError:
        logger.error(f"Training dataset not found in {data_dir}.")
        raise
    
    try:
        validation_dataset = torch.load(data_dir / "validation_dataset.pt", weights_only=False)
        logger.info("Validation dataset loaded.")
    except FileNotFoundError:
        logger.error(f"Validation dataset not found in {data_dir}.")
        raise
        
    try:
        test_dataset = torch.load(data_dir / "test_dataset.pt", weights_only=False)
        logger.info("Test dataset loaded.")
    except FileNotFoundError:
        logger.error(f"Test dataset not found in {data_dir}.")
        raise
        
    return training_dataset, validation_dataset, test_dataset


def train_tft_model(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    model_output_dir: Path,
    log_dir: Path,
    logger: logging.Logger,
    hidden_size: int = 32,
    lstm_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    patience: int = 10,
    max_epochs: int = 100,
    gpus: int = 1 if torch.cuda.is_available() else 0,
    gradient_clip_val: float = 0.1,
    gradient_clip_algorithm: str = "norm",
    use_swa: bool = False,
    swa_learning_rate: float = 0.05,
    use_onecycle: bool = False,
    ckpt_path: str = None
) -> None:
    """
    Configure and train the TFT model.

    Args:
        training_dataset: Training TimeSeriesDataSet.
        validation_dataset: Validation TimeSeriesDataSet.
        model_output_dir: Directory to save the trained model and checkpoints.
        log_dir: Directory for TensorBoard logs.
        logger: Logger instance.
        hidden_size: Hidden size of network layers.
        lstm_layers: Number of LSTM layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        learning_rate: Initial learning rate.
        patience: Patience for early stopping.
        max_epochs: Maximum number of training epochs.
        gpus: Number of GPUs to use (0 for CPU).
        gradient_clip_val: Gradient clipping value.
        gradient_clip_algorithm: Gradient clipping algorithm ("norm" or "value").
        use_swa: Whether to use Stochastic Weight Averaging.
        swa_learning_rate: Learning rate for SWA.
        use_onecycle: Whether to use OneCycleLR learning rate scheduler.
        ckpt_path: Path to checkpoint to resume training from.
    """
    logger.info("Starting TFT model training...")

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=patience, 
        verbose=True, 
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks_list = [lr_monitor, early_stop_callback]
    if use_swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=swa_learning_rate)
        callbacks_list.append(swa_callback)
        logger.info(f"Stochastic Weight Averaging (SWA) enabled with SWA LR: {swa_learning_rate}")

    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tft_model", log_graph=True)

    class TFTWithScheduler(TemporalFusionTransformer):
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            scheduler = {
                'scheduler': lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-8),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return [opt], [scheduler]

    tft_model = TFTWithScheduler.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=num_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        lstm_layers=lstm_layers,
        output_size=11,  # Number of quantiles to predict (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
        loss=QuantileLoss(),
        log_interval=50, # Log N times per epoch
        reduce_on_plateau_patience=4,
    )
    logger.info(f"Temporal Fusion Transformer model initialized:")

    lr_scheduler_config = None
    if use_onecycle:
        logger.info("Using OneCycleLR scheduler.")
        steps_per_epoch = len(training_dataset) // args.batch_size
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "scheduler_args": {
                "max_lr": learning_rate,
                "steps_per_epoch": steps_per_epoch,
                "epochs": max_epochs,
                "anneal_strategy": "cos",
            },
        }

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 'auto',
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        callbacks=callbacks_list,
        logger=tb_logger,
        default_root_dir=str(model_output_dir),
        precision=args.precision,
        benchmark=args.benchmark,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler=args.profiler,
    )

    logger.info("Training the model...")
    if use_onecycle and lr_scheduler_config is not None:
        optimizer = tft_model.configure_optimizers()[0]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            anneal_strategy="cos"
        )
        trainer.fit(
            tft_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )
    else:
        trainer.fit(
            tft_model,
            train_dataloaders=training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers),
            val_dataloaders=validation_dataset.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers),
            ckpt_path=ckpt_path
        )

    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model path: {best_model_path}")
    best_tft_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    final_model_save_path = model_output_dir / "best_tft_model.ckpt"
    trainer.save_checkpoint(final_model_save_path)
    logger.info(f"Best model saved to {final_model_save_path}")

    logger.info("TFT model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "data/prepared/tft_datasets")),
        help="Directory where preprocessed datasets are stored.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "models/tft_model")),
        help="Directory to save the trained model and checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "logs/tft_training")),
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size of network layers.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=50, help="Patience for early stopping.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0, help="Number of GPUs to use (0 for CPU).")
    parser.add_argument("--gradient-clip-val", type=float, default=0.1, help="Gradient clipping value.")
    parser.add_argument(
        "--precision", 
        type=str, 
        default="32-true", 
        choices=["32-true", "16-mixed", "bf16-mixed", "64-true"],
        help="Training precision (e.g., 32-true, 16-mixed, bf16-mixed)."
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Enable cudnn.benchmark for potential speedup if input sizes are constant."
    )
    parser.add_argument(
        "--deterministic", 
        action="store_true", 
        help="Enable deterministic mode for reproducibility. Might slow down training."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility when deterministic is True."
    )
    parser.add_argument(
        "--accumulate-grad-batches", 
        type=int, 
        default=1, 
        help="Accumulates gradients over k batches before stepping the optimizer."
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced"],
        help="Enable PyTorch Lightning profiler (e.g., simple, advanced). Output in log_dir."
    )
    parser.add_argument(
        "--auto-lr-find",
        action="store_true",
        help="Enable automatic learning rate finding before training."
    )
    parser.add_argument(
        "--gradient-clip-algorithm",
        type=str,
        default="norm",
        choices=["norm", "value"],
        help="Gradient clipping algorithm to use ('norm' or 'value'). Default is 'norm'."
    )
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Enable Stochastic Weight Averaging (SWA)."
    )
    parser.add_argument(
        "--swa-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for SWA scheduler. Default is 0.05."
    )
    parser.add_argument(
        "--use-onecycle",
        action="store_true",
        help="Use OneCycleLR learning rate scheduler."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )

    args = parser.parse_args()
    
    if args.deterministic:
        seed_everything(args.seed, workers=True)

    logger = setup_logger(name=__name__, level=args.log_level.upper())

    data_dir = Path(args.data_dir)
    model_output_dir = Path(args.model_output_dir)
    log_dir = Path(args.log_dir)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        training_dataset, validation_dataset, test_dataset = load_datasets(data_dir, logger)
    except FileNotFoundError:
        logger.error("Essential dataset(s) not found during initial load. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during initial dataset load: {e}", exc_info=True)
        sys.exit(1)

    temp_model_for_lr_find = None
    tuned_lr = args.learning_rate

    if args.auto_lr_find:
        logger.info("Running Automatic Learning Rate Finder...")
        
        temp_model_for_lr_find = TemporalFusionTransformer.from_dataset(
            training_dataset, # Use actual training_dataset for correct initialization
            learning_rate=args.learning_rate, # Placeholder, will be tuned
            hidden_size=args.hidden_size,
            attention_head_size=args.num_heads,
            dropout=args.dropout,
            hidden_continuous_size=args.hidden_size // 2,
            lstm_layers=args.lstm_layers,
            output_size=7, 
            loss=QuantileLoss(),
            log_interval=50, 
            reduce_on_plateau_patience=4,
        )

        temp_trainer_for_lr_find = L.Trainer(
            max_epochs=args.max_epochs, # Will be overridden by tune
            accelerator='gpu' if args.gpus > 0 else 'cpu',
            devices=args.gpus if args.gpus > 0 else 'auto',
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            logger=False, # Disable logging for LR find
            enable_checkpointing=False, # Disable checkpointing for LR find
            enable_progress_bar=False, # Disable progress bar for LR find
            enable_model_summary=False, # Disable model summary for LR find
            precision=args.precision,
            benchmark=args.benchmark,
            deterministic=args.deterministic,
            accumulate_grad_batches=args.accumulate_grad_batches,
            profiler=None # Disable profiler for LR find
        )
        
        tuner = Tuner(temp_trainer_for_lr_find)
        lr_finder_result = tuner.lr_find(
            temp_model_for_lr_find, 
            train_dataloaders=training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers),
            min_lr=1e-8, 
            max_lr=1.0, 
            num_training=100
        )
        
        if lr_finder_result and hasattr(lr_finder_result, 'suggestion') and lr_finder_result.suggestion() is not None:
            tuned_lr = lr_finder_result.suggestion()
            logger.info(f"Auto LR Finder suggested learning rate: {tuned_lr}")
        else:
            logger.warning("LR Finder did not suggest a learning rate. Using default or provided LR.")
            tuned_lr = args.learning_rate # Fallback to originally provided LR
        if temp_model_for_lr_find is not None:
            del temp_model_for_lr_find
        del temp_trainer_for_lr_find
        del tuner

    try:
        train_tft_model(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            model_output_dir=model_output_dir,
            log_dir=log_dir,
            logger=logger,
            hidden_size=args.hidden_size,
            lstm_layers=args.lstm_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            learning_rate=tuned_lr,
            patience=args.patience,
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            use_swa=args.use_swa,
            swa_learning_rate=args.swa_learning_rate,
            use_onecycle=args.use_onecycle,
            ckpt_path=args.ckpt_path
        )
    except FileNotFoundError:
        logger.error("Essential dataset(s) not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1) 