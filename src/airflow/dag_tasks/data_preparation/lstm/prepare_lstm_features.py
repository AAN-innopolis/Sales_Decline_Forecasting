"""
Script for preparing features for LSTM model for predicting sales reduction.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.utils.data_utils import setup_logger
from src.features.temporal_features import (
    get_rolling_features,
    get_lag_features
)
from src.config.configs import settings


def prepare_lstm_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    The function for preparing data for LSTM model.
    
    Args:
        df_original (pd.DataFrame): DataFrame with base features
        logger (logging.Logger): The logger object
        
    Returns:
        pd.DataFrame: DataFrame with features for LSTM
    """
    df = df_original.copy().sort_values(by=['store', 'date'])
    logger.info("Starting preparation of features for LSTM model")
    
    df_lag = get_lag_features(df, logger) 
    df_rolling = get_rolling_features(df, logger)
    df_res = df.merge(
        df_lag, 
        on=['store', 'date'], 
        how='left'
    ).merge(
        df_rolling, 
        on=['store', 'date'], 
        how='left'
    )
    logger.info(f"Final data shape: {df_res.shape}")
    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The pipeline for preparing the features for LSTM model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/lstm_tft_features.parquet',
                        help='The path to the file with lstm_tft features')
    parser.add_argument('--output-file', type=str, default='data/prepared/lstm_features.parquet',
                        help='The path to save the features for LSTM model')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Lstm_tft features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading lstm_tft features: {e}")
    
    df_lstm = prepare_lstm_data(df, logger)

    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        df_lstm.to_parquet(Path(settings.PROJECT_ROOT, args.output_file))
        logger.info(f"Lstm features saved in {Path(settings.PROJECT_ROOT, args.output_file)}")
    except Exception as e:
        raise Exception(f"Error while saving lstm features: {e}")
    
