"""
Script for preparing the features for LSTM and TFT models.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.utils.data_utils import setup_logger, load_data
from src.config.configs import settings
from src.features.transaction_aggregation_features import (
    get_base_statistics,
    get_extended_statistics,
)
from src.features.static_features import ( 
    get_cyclical_features
)
from src.features.store_date_features import (
    get_derived_features
)


def prepare_lstm_tft_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    The function for preparing the features for LSTM and TFT models.
    
    Args:
        df_original (pd.DataFrame): The source DataFrame
        logger (logging.Logger): The logger object
        
    Returns:
        pd.DataFrame: The DataFrame with the features for LSTM and TFT models
    """
    logger.info("Starting preparation of the features for LSTM and TFT models")
    logger.info(f"Original data shape: {df_original.shape}")
    df = df_original.copy()

    df_base = get_base_statistics(df, logger)
    df_extended = get_extended_statistics(df, logger)
    df_cyclical = get_cyclical_features(df, logger)
    
    
    df_res = (
        df_base.merge(
            df_extended, 
            on=['store', 'date'], 
            how='left'
        ).merge(
            df_cyclical, 
            on=['date'], 
            how='left'
        )
    )
    df_derived = get_derived_features(df_res, logger)
    df_res = df_res.merge(
            df_derived, 
            on=['store', 'date'], 
            how='left'
        )
    logger.info(f"Final data shape: {df_res.shape}")
    return df_res


if __name__ == "__main__":
    """
    The main function for preparing the features for LSTM and TFT models
    """
    parser = argparse.ArgumentParser(description='The pipeline for data engineering for LSTM and TFT models')
    parser.add_argument('--input-file', type=str, default='data/prepared/cleaned_data.parquet',
                        help='The path to the cleaned data file')
    parser.add_argument('--output-file', type=str, default='data/prepared/lstm_tft_features.parquet',
                        help='The path to save the features for LSTM and TFT models')
    parser.add_argument('--log-level', 
                        type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO',
                        help='The logging level')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading cleaned data: {e}")
    
    df_lstm_tft = prepare_lstm_tft_features(df, logger)
    
    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        lstm_tft_features_path = Path(settings.PROJECT_ROOT, args.output_file)
        df_lstm_tft.to_parquet(lstm_tft_features_path)
        logger.info(f"LSTM and TFT features saved in {lstm_tft_features_path}")
    except Exception as e:
        raise Exception(f"Error while saving LSTM and TFT features: {e}")
