"""
Script for preparing features for TFT model for predicting sales reduction.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.features.temporal_features import get_rolling_features
from src.utils.data_utils import setup_logger
from src.features.static_features import (
    get_holiday_features,
    get_store_attributes,
)
from src.features.store_date_features import get_store_features
from src.config.configs import settings


def prepare_tft_data(
    df_original: pd.DataFrame,
    df_transactions: pd.DataFrame,
    logger: logging.Logger
) -> tuple:
    """
    Preparation of data for TFT model.
    
    Args:
        df_original (pd.DataFrame): DataFrame with original features
        df_transactions (pd.DataFrame): DataFrame with transactions level information
        logger (logging.Logger): Logger object
        
    Returns:
        pd.DataFrame with TFT features
    """
    df = df_original.copy()
    logger.info("Starting preparation of TFT features")
    
    df_holidays = get_holiday_features(df, logger)
    df_store_attributes = get_store_attributes(df_transactions, logger)

    df_res = (df
        .merge(
            df_holidays, 
            on=['date'], 
            how='left'
        ).merge(
            df_store_attributes, 
            on=['store'], 
            how='left'
        )
    )
    df_store_features = get_store_features(df_res, logger)
    df_rolling = get_rolling_features(df, logger) 

    df_res = (df_res
        .merge(
            df_store_features, 
            on=['store', 'date'], 
            how='left'
        ).merge(
            df_rolling, 
            on=['store', 'date'], 
            how='left'
        )
    )
    df_res['time_idx'] = df_res.groupby('store').cumcount()

    for col in df_res.columns:
        if df_res[col].isna().any():
            logger.warning(f"Column {col} has NaN values: {df_res[col].isna().sum()}")
            df_res[col] = df_res[col].fillna(-1)

    logger.info(f"Final data shape: {df_res.shape}")
    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparation of features for TFT model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/lstm_tft_features.parquet',
                        help='Path to file with lstm_tft features')
    parser.add_argument('--transactions', type=str, default='data/prepared/cleaned_data.parquet',
                        help='Path to file with cleaned data')
    parser.add_argument('--output-file', type=str, default='data/prepared/tft_features.parquet',
                        help='Path to file for saving tft features')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        df_transactions = pd.read_parquet(Path(settings.PROJECT_ROOT, args.transactions))
        logger.info(f"Original features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error loading original features: {e}")
    
    df_tft = prepare_tft_data(df, df_transactions, logger)

    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        df_tft.to_parquet(Path(settings.PROJECT_ROOT, args.output_file))
        
        logger.info(f"TFT features saved in {Path(settings.PROJECT_ROOT, args.output_file)}")
    except Exception as e:
        raise Exception(f"Error saving TFT features: {e}")
    