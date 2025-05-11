"""
Script for encoding features for Temporal Fusion Transformer model.
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
from src.utils.data_utils.tft_feature_encoder import TFTFeatureEncoder
from src.config.configs import settings


def encode_tft_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        max_encoder_length: int = 90,
        max_prediction_length: int = 30,
        target_col: str = 'purchase_amount'
    ) -> tuple:
    """
    Encode features for Temporal Fusion Transformer model.
    
    Args:
        df_original: Input DataFrame with TFT features
        logger: Logger instance
        max_encoder_length: Maximum lookback window length
        max_prediction_length: Maximum prediction horizon
        target_col: Target column for prediction
        
    Returns:
        tuple: (DataFrame with encoded features, TFT dataset parameters)
    """
    df = df_original.copy()
    logger.info(f"Encoding features for TFT model with target {target_col}")
    models_dir = Path(settings.PROJECT_ROOT, 'models/embeddings')
    models_dir.mkdir(exist_ok=True, parents=True)
    
    feature_categories = {
        # Static features - unchanging properties of the series
        'static_categoricals': [],
        'static_reals': [],
        
        # Time-varying known features - values known in advance
        'time_varying_known_categoricals': [],
        'time_varying_known_reals': [
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'quarter_sin', 'quarter_cos'
        ],
        
        # Time-varying unknown features - values only known at prediction time
        'time_varying_unknown_categoricals': [],
        'time_varying_unknown_reals': [
            target_col,
            'purchased_bottles', 
            'purchased_liters', 
            'transaction_count',
            'unique_categories',
            'unique_items',
            'avg_price_per_bottle',
            'avg_price_per_liter',
            'avg_transaction_value'
        ]
    }
    
    # Add store location attributes if available
    location_features = {'city', 'county', 'zipcode'}
    for feature in location_features:
        if feature in df.columns:
            feature_categories['static_categoricals'].append(feature)
    
    # Add store geographical coordinates if available
    for feature in ['lat', 'lon']:
        if feature in df.columns:
            feature_categories['static_reals'].append(feature)
    
    # Add holiday-related features if available
    for feature in ['is_holiday', 'holiday_name']:
        if feature in df.columns:
            feature_categories['time_varying_known_categoricals'].append(feature)
    
    if 'days_to_nearest_holiday' in df.columns:
        feature_categories['time_varying_known_reals'].append('days_to_nearest_holiday')
    
    # Add lag and rolling features to unknown reals
    time_series_features = [col for col in df.columns if any(pattern in col for pattern in ['lag_', 'roll_', 'days_since_'])]
    feature_categories['time_varying_unknown_reals'].extend(time_series_features)
    
    tft_encoder = TFTFeatureEncoder(
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_col=target_col,
        group_ids=['store'],
        **feature_categories,
        logger=logger
    )
    
    df_encoded = tft_encoder.fit_transform(df)
    tft_encoder.save(models_dir / 'tft_feature_encoder.pkl')
    
    tft_params = tft_encoder.create_tft_dataset_parameters()
    
    logger.info(f"Encoding completed. \n\
                \rDataFrame shape: {df_encoded.shape}\n\
                \rTFT parameters: {tft_params}\n\
                \rColumns: {df_encoded.columns}")
    return df_encoded, tft_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for encoding features for TFT model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/tft_features.parquet',
                        help='Path to the input file with TFT features')
    parser.add_argument('--output-file', type=str, default='data/prepared/tft_features_encoded.parquet',
                        help='Path to save the encoded TFT features')
    parser.add_argument('--params-file', type=str, default='data/prepared/tft_params.pkl',
                        help='Path to save the TFT parameters')
    parser.add_argument('--max-encoder-length', type=int, default=90,
                        help='Maximum encoder length (lookback window)')
    parser.add_argument('--max-prediction-length', type=int, default=30,
                        help='Maximum prediction length (forecast horizon)')
    parser.add_argument('--target-col', type=str, default='purchase_amount',
                        help='Target column for prediction')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"TFT features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading TFT features: {e}")
    
    df_encoded, tft_params = encode_tft_data(
        df, 
        logger=logger,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        target_col=args.target_col
    )
    
    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        df_encoded.to_parquet(Path(settings.PROJECT_ROOT, args.output_file))
        logger.info(f"Encoded TFT features saved to {Path(settings.PROJECT_ROOT, args.output_file)}")
    except Exception as e:
        raise Exception(f"Error while saving encoded TFT features: {e}")
    
    try:
        params_dir = Path(settings.PROJECT_ROOT, args.params_file).parent
        params_dir.mkdir(exist_ok=True, parents=True)
        
        import pickle
        with open(Path(settings.PROJECT_ROOT, args.params_file), 'wb') as f:
            pickle.dump(tft_params, f)
            
        logger.info(f"TFT parameters saved to {Path(settings.PROJECT_ROOT, args.params_file)}")
    except Exception as e:
        raise Exception(f"Error while saving TFT parameters: {e}") 