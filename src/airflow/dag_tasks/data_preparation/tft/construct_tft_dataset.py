"""
Script for constructing TFT datasets with train/validation/test splits.
"""

import argparse
from math import floor
import pandas as pd
from pathlib import Path
import sys
import logging
import pickle
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.utils.data_utils import setup_logger
from src.config.configs import settings


def split_data_by_store(
    df: pd.DataFrame,
    split_ratio: float,
    logger: logging.Logger
) -> tuple:
    """
    Split data into train, validation and test sets 
    by time for each store.
    
    Args:
        df: Input DataFrame
        split_ratio: Ratio of data for splitting
        logger: Logger instance
        
    Returns:
        tuple: (train_df, val_df)
    """
    train_dfs = []
    val_dfs = []
    
    logger.info(f"Data split by time for each store:")
    for _, store_df in df.groupby('store'):
        train_end = int(len(store_df) * split_ratio)
        train_dfs.append(store_df.iloc[:train_end])
        val_dfs.append(store_df.iloc[train_end:])
    # Concatenate splits. Stores will be distinguished 
    # by group_ids parameter in TimeSeriesDataSet
    train_df = pd.concat(train_dfs, axis=0)
    val_df = pd.concat(val_dfs, axis=0)

    logger.info(f"Train: {len(train_df)} rows\n\
                \rValidation: {len(val_df)} rows\n\
                \rStore coverage: {train_df['store'].nunique()}")
    return train_df, val_df


def construct_tft_datasets(
    df: pd.DataFrame,
    logger: logging.Logger,
    min_history_length: int = 50,
    target_col: str = 'purchase_amount',
    split_ratio: float = 0.7,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Construct and save TFT datasets.
    
    Args:
        df: Input DataFrame with encoded features
        logger: Logger instance
        min_history_length: Minimum history length
        target_col: Target column name
        split_ratio: Ratio of data for splitting
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Starting construction of TFT datasets")
    feature_categories = {
        # Static features - unchanging properties of the series
        'static_categoricals': [
            'store','name','address','city','zipcode','county'
        ],
        'static_reals': ['lon','lat'],
        # Time-varying known features - values known in advance
        'time_varying_known_categoricals': ['is_holiday', 'holiday_name'],
        'time_varying_known_reals': [
            *[col for col in df.columns if col.startswith('day')],
            *[col for col in df.columns if col.startswith('month')],
            *[col for col in df.columns if col.startswith('quarter')],
            *[col for col in df.columns if col.startswith('week')],
            *[col for col in df.columns if col.startswith('year')],
        ],
    }
    feature_categories['time_varying_unknown_reals'] = (
        set(df.columns) 
        - set().union(*[set(v) for v in feature_categories.values()])
        - {'date', 'time_idx'}
    )
    # Initialize and pre-fit encoders on the entire dataset
    logger.info("Initializing and pre-fitting categorical encoders on the entire dataset...")
    categorical_encoders_dict = {
        **{
            col: NaNLabelEncoder(add_nan=True) 
            for col in feature_categories['static_categoricals']
        },
        **{
            col: NaNLabelEncoder(add_nan=True) 
            for col in feature_categories['time_varying_known_categoricals']
        }
    }
    for col_name, encoder in categorical_encoders_dict.items():
        df[col_name] = df[col_name].astype('str')
        unique_values = df[col_name].unique()
        encoder.fit(unique_values)
    logger.info("Categorical encoders pre-fitted.")
    
    # Filter stores with sufficient number of records
    store_counts = df.groupby('store').size()
    valid_stores = store_counts[store_counts >= min_history_length].index
    logger.info(f"Filtered out stores with insufficient records: \n\
                \r{store_counts[store_counts < min_history_length].shape[0]} stores. \n\
                \rRemaining stores: {len(valid_stores)}")

    df['purchase_amount'] = df['purchase_amount'].clip(lower=0)
    train_df, val_df = split_data_by_store(
        df[df['store'].isin(valid_stores)].copy(), 
        split_ratio, 
        logger
    )
    test_df = df[~df['store'].isin(valid_stores)].copy()
    logger.info(f"Test set: {len(test_df)} rows")

    min_history_length = floor(min(
        min_history_length*split_ratio, 
        min_history_length*(1-split_ratio)
        )) // 5
    logger.info(f"Minimum history length: {min_history_length * 2}")
    logger.info("Constructing training dataset...")
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["store"],
        min_encoder_length=min_history_length//2,
        max_encoder_length=min_history_length,
        min_prediction_length=min_history_length//2,
        max_prediction_length=min_history_length,
        static_categoricals=feature_categories['static_categoricals'],
        static_reals=feature_categories['static_reals'],
        time_varying_known_categoricals=feature_categories['time_varying_known_categoricals'],
        time_varying_known_reals=feature_categories['time_varying_known_reals'],
        time_varying_unknown_reals=feature_categories['time_varying_unknown_reals'],
        target_normalizer=GroupNormalizer(
            groups=["store"], 
            transformation="softplus"
        ),
        categorical_encoders=categorical_encoders_dict,
        randomize_length=True,
        add_relative_time_idx=True,
        # add_target_scales=True,
        # add_encoder_length=True,
    )
    logger.info("Training dataset constructed.")
    logger.info("Constructing validation dataset...")
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        val_df, 
        predict=True, 
        stop_randomization=True,
    )
    logger.info("Validation dataset constructed.")
    logger.info("Constructing test dataset...")
    test = TimeSeriesDataSet.from_dataset(
        training, 
        test_df, 
        predict=True, 
        stop_randomization=True
    )
    logger.info("Test dataset constructed.")
    return training, validation, test
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for constructing TFT datasets')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/tft_features.parquet',
                        help='Path to the input file with TFT features')
    parser.add_argument('--output-dir', type=str, default='data/prepared/tft_datasets',
                        help='Path to the output directory for TFT datasets')
    parser.add_argument('--min-history-length', type=int, default=120,
                        help='Minimum history length')
    parser.add_argument('--target-col', type=str, default='purchase_amount',
                        help='Target column name')
    parser.add_argument('--split-ratio', type=float, default=0.7,
                        help='Ratio of data for splitting')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Prepared TFT features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading prepared TFT features: {e}")
    
    train, val, test = construct_tft_datasets(
        df=df,
        logger=logger,
        min_history_length=args.min_history_length,
        target_col=args.target_col,
        split_ratio=args.split_ratio,
    ) 
    try:
        datasets_dir = Path(settings.PROJECT_ROOT, args.output_dir)
        datasets_dir.mkdir(exist_ok=True, parents=True)

        torch.save(train, datasets_dir / 'training_dataset.pt')
        torch.save(val, datasets_dir / 'validation_dataset.pt')
        torch.save(test, datasets_dir / 'test_dataset.pt')
        logger.info(f"TFT datasets saved to {datasets_dir}: \n\
                    \rTraining: {datasets_dir / 'training_dataset.pt'}\n\
                    \rValidation: {datasets_dir / 'validation_dataset.pt'}\n\
                    \rTest: {datasets_dir / 'test_dataset.pt'}")
    except Exception as e:
        raise Exception(f"Error while saving TFT datasets: {e}")
