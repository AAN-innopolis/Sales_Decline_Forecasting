"""
Script for constructing LSTM datasets with train/validation/test splits.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.utils.data_utils import setup_logger
from src.features.lstm_dataset import create_lstm_datasets
from src.config.configs import settings


def construct_lstm_datasets(
    df: pd.DataFrame,
    logger: logging.Logger,
    sequence_length: int = 30,
    prediction_length: int = 30,
    target_col: str = 'purchase_amount',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    min_store_history: int = 90,
    embedding_size: int = 16
) -> None:
    """
    Construct and save LSTM datasets.
    
    Args:
        df: Input DataFrame with features
        logger: Logger instance
        sequence_length: Length of input sequences
        prediction_length: Length of prediction horizon
        target_col: Target column name
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        batch_size: Batch size for DataLoader
        min_store_history: Minimum number of historical points required for a store
    """
    logger.info("Starting construction of LSTM datasets")
    
    # Define feature columns
    feature_cols = [
        # Time features
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        
        # Rolling statistics
        f'hist_mean_{sequence_length}D_purchases_amount',
        f'hist_std_{sequence_length}D_purchases_amount',
        f'hist_max_{sequence_length}D_purchases_amount',
        f'hist_min_{sequence_length}D_purchases_amount',
        f'hist_median_{sequence_length}D_purchases_amount',
        
        # Momentum features
        f'purchase_momentum_{sequence_length}D',
        f'purchase_momentum_pct_{sequence_length}D',

        # Store embeddings
        *[f'store_emb_{emb}' for emb in range(embedding_size)]
    ]
    
    train_loader, val_loader, test_loader = create_lstm_datasets(
        df=df,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        target_col=target_col,
        feature_cols=feature_cols,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        min_store_history=min_store_history,
        logger=logger
    )
    
    # Save datasets
    datasets_dir = Path(settings.PROJECT_ROOT, 'data/prepared/lstm_datasets')
    datasets_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(train_loader, datasets_dir / 'train_loader.pt')
    torch.save(val_loader, datasets_dir / 'val_loader.pt')
    torch.save(test_loader, datasets_dir / 'test_loader.pt')
    
    logger.info(f"LSTM datasets saved in {datasets_dir}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for constructing LSTM datasets')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/lstm_features_with_embeddings.parquet',
                        help='Path to the input file with LSTM features')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Length of input sequences')
    parser.add_argument('--prediction-length', type=int, default=30,
                        help='Length of prediction horizon')
    parser.add_argument('--target-col', type=str, default='purchase_amount',
                        help='Target column name')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Ratio of data for testing')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for DataLoader')
    parser.add_argument('--min-store-history', type=int, default=90,
                        help='Minimum number of historical points required for a store')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"LSTM features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading LSTM features: {e}")
    
    construct_lstm_datasets(
        df=df,
        logger=logger,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
        target_col=args.target_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        min_store_history=args.min_store_history
    ) 