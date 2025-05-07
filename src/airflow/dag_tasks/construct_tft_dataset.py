"""
Script for constructing TFT datasets with train/validation/test splits.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os
import pickle
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import setup_logger
from src.config.configs import settings


def split_data_by_store(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    logger: logging.Logger
) -> tuple:
    """
    Split data into train, validation and test sets by time for each store.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        logger: Logger instance
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Sort data by store and time_idx
    df = df.sort_values(['store', 'time_idx'])
    
    # Initialize empty DataFrames for each split
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split data for each store
    for store in df['store'].unique():
        store_data = df[df['store'] == store].copy()
        n_samples = len(store_data)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        train_dfs.append(store_data.iloc[:train_end])
        val_dfs.append(store_data.iloc[train_end:val_end])
        test_dfs.append(store_data.iloc[val_end:])
    
    # Combine splits
    train_df = pd.concat(train_dfs, axis=0)
    val_df = pd.concat(val_dfs, axis=0)
    test_df = pd.concat(test_dfs, axis=0)
    
    # Log split information
    logger.info(f"Data split by time for each store:")
    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Validation: {len(val_df)} rows")
    logger.info(f"Test: {len(test_df)} rows")
    
    # Log store coverage
    train_stores = set(train_df['store'].unique())
    val_stores = set(val_df['store'].unique())
    test_stores = set(test_df['store'].unique())
    all_stores = set(df['store'].unique())
    
    logger.info(f"Store coverage:")
    logger.info(f"Train: {len(train_stores)}/{len(all_stores)} stores")
    logger.info(f"Validation: {len(val_stores)}/{len(all_stores)} stores")
    logger.info(f"Test: {len(test_stores)}/{len(all_stores)} stores")
    
    return train_df, val_df, test_df


def construct_tft_datasets(
    df: pd.DataFrame,
    logger: logging.Logger,
    max_encoder_length: int = 90,
    max_prediction_length: int = 30,
    target_col: str = 'purchase_amount',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32
) -> None:
    """
    Construct and save TFT datasets.
    
    Args:
        df: Input DataFrame with encoded features
        logger: Logger instance
        max_encoder_length: Maximum lookback window length
        max_prediction_length: Maximum prediction horizon
        target_col: Target column name
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        batch_size: Batch size for DataLoader
    """
    logger.info("Starting construction of TFT datasets")
    
    # Split data by store
    train_df, val_df, test_df = split_data_by_store(
        df, train_ratio, val_ratio, test_ratio, logger
    )
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["store"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['store', 'city', 'county', 'zipcode'],
        static_reals=['lat', 'lon'],
        time_varying_known_categoricals=['is_holiday', 'holiday_name'],
        time_varying_known_reals=[
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'quarter_sin', 'quarter_cos',
            'days_to_nearest_holiday'
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            target_col,
            'purchased_bottles', 
            'purchased_liters', 
            'transaction_count',
            'unique_categories',
            'unique_items',
            'avg_price_per_bottle',
            'avg_price_per_liter',
            'avg_transaction_value'
        ] + [col for col in df.columns if any(pattern in col for pattern in ['lag_', 'roll_', 'days_since_'])],
        target_normalizer=GroupNormalizer(
            groups=["store"], 
            transformation="softplus"
        ),
         categorical_encoders={
            'store': NaNLabelEncoder(add_nan=True),
            'city': NaNLabelEncoder(add_nan=True),
            'county': NaNLabelEncoder(add_nan=True),
            'zipcode': NaNLabelEncoder(add_nan=True),
            'holiday_name': NaNLabelEncoder(add_nan=True),
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        val_df, 
        predict=True, 
        stop_randomization=True,
    )
    
    # Create test dataset
    test = TimeSeriesDataSet.from_dataset(
        training, 
        test_df, 
        predict=True, 
        stop_randomization=True
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=4
    )
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=4
    )
    test_dataloader = test.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=4
    )
    
    # Save datasets and parameters
    datasets_dir = Path(settings.PROJECT_ROOT, 'data/prepared/tft_datasets')
    datasets_dir.mkdir(exist_ok=True, parents=True)
    
    # Save datasets
    torch.save(training, datasets_dir / 'training_dataset.pt')
    torch.save(validation, datasets_dir / 'validation_dataset.pt')
    torch.save(test, datasets_dir / 'test_dataset.pt')
    
    # Save dataloaders
    torch.save(train_dataloader, datasets_dir / 'train_dataloader.pt')
    torch.save(val_dataloader, datasets_dir / 'val_dataloader.pt')
    torch.save(test_dataloader, datasets_dir / 'test_dataloader.pt')
    
    # Save TFT parameters
    tft_params = {
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "time_idx": "time_idx",
        "target": target_col,
        "group_ids": ["store"],
        "static_categoricals": ['store', 'city', 'county', 'zipcode'],
        "static_reals": ['lat', 'lon'],
        "time_varying_known_categoricals": ['is_weekend', 'is_holiday', 'holiday_name'],
        "time_varying_known_reals": [
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'quarter_sin', 'quarter_cos',
            'days_to_nearest_holiday'
        ],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [
            target_col,
            'purchased_bottles', 
            'purchased_liters', 
            'transaction_count',
            'unique_categories',
            'unique_items',
            'avg_price_per_bottle',
            'avg_price_per_liter',
            'avg_transaction_value'
        ] + [col for col in df.columns if any(pattern in col for pattern in ['lag_', 'roll_', 'days_since_'])]
    }
    
    with open(datasets_dir / 'tft_params.pkl', 'wb') as f:
        pickle.dump(tft_params, f)
    
    logger.info(f"TFT datasets saved in {datasets_dir}")
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")
    logger.info(f"Test batches: {len(test_dataloader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for constructing TFT datasets')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/tft_features_encoded.parquet',
                        help='Path to the input file with encoded TFT features')
    parser.add_argument('--max-encoder-length', type=int, default=10,
                        help='Maximum encoder length (lookback window)')
    parser.add_argument('--max-prediction-length', type=int, default=10,
                        help='Maximum prediction length (forecast horizon)')
    parser.add_argument('--target-col', type=str, default='purchase_amount',
                        help='Target column name')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Ratio of data for testing')
    parser.add_argument('--batch-size', type=int, default=80,
                        help='Batch size for DataLoader')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        df[['store', 'zipcode','is_holiday']] = df[['store', 'zipcode','is_holiday']].astype(str)
        logger.info(f"Encoded TFT features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading encoded TFT features: {e}")
    
    construct_tft_datasets(
        df=df,
        logger=logger,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        target_col=args.target_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size
    ) 