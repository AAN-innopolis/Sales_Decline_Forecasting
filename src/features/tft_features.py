"""
TFT-specific feature engineering module.
Contains functions for creating features specifically for Temporal Fusion Transformer models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import setup_logger
from features.core_features import get_store_attributes, get_holiday_features


def create_tft_features(
        df: pd.DataFrame, 
        logger: logging.Logger, 
        target: str = 'sale_dollars'
    ) -> pd.DataFrame:
    """
    Create features specifically for TFT models.
    
    Args:
        df (pd.DataFrame): Input dataframe with base features
        logger: Logger instance
        target (str): Target variable for prediction
        
    Returns:
        pd.DataFrame: Dataframe with TFT-specific features
    """
    logger.info(f"Creating TFT-specific features for target {target}")
    
    df_features = df.copy()
    
    # Define initial feature groups for TFT
    static_categoricals = ['store']
    static_reals = []
    
    time_varying_known_categoricals = ['is_weekend']
    time_varying_known_reals = [
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos'
    ]
    
    # Add store attributes if the necessary columns exist in the dataframe
    try:
        # Get store attributes safely
        store_attributes = get_store_attributes(df)
        df_features = pd.merge(df_features, store_attributes, on=['store'], how='left')
        
        # Add the columns to feature groups if they exist
        if 'city' in df_features.columns:
            static_categoricals.append('city')
        if 'county' in df_features.columns:
            static_categoricals.append('county')
        if 'zipcode' in df_features.columns:
            static_categoricals.append('zipcode')
        if 'lon' in df_features.columns:
            static_reals.append('lon')
        if 'lat' in df_features.columns:
            static_reals.append('lat')
            
    except Exception as e:
        logger.warning(f"Could not get store attributes: {e}")
        logger.warning("Continuing without store attributes")
    
    # Try to add holiday features if possible
    try:
        # Get holiday features
        holiday_features = get_holiday_features(df_features)
        df_features = df_features.join(holiday_features)
        
        # Add holiday features to the appropriate groups if they exist
        if 'is_holiday' in df_features.columns:
            time_varying_known_categoricals.append('is_holiday')
        if 'holiday_name' in df_features.columns:
            time_varying_known_categoricals.append('holiday_name')
        if 'days_to_nearest_holiday' in df_features.columns:
            time_varying_known_reals.append('days_to_nearest_holiday')
    
    except Exception as e:
        logger.warning(f"Could not get holiday features: {e}")
        logger.warning("Continuing without holiday features")
    
    # Time-varying variables that are unknown in the future
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = [
        target,
        'sale_bottles', 
        'sale_liters', 
        'transaction_count',
        'unique_categories',
        'unique_items'
    ]
    
    # Add time-varying features if they exist
    if 'days_since_prev_purchase' in df_features.columns:
        time_varying_unknown_reals.append('days_since_prev_purchase')
    
    # Add lag features for relevant metrics
    lag_windows = [1, 2, 3, 7, 14, 28]
    for lag in lag_windows:
        for col in [target, 'transaction_count']:
            lag_name = f'{col}_lag_{lag}'
            df_features[lag_name] = df_features.groupby('store')[col].shift(lag)
            time_varying_unknown_reals.append(lag_name)
    
    # Add rolling window statistics
    window_sizes = [7, 14, 30, 60, 90]
    for window in window_sizes:
        # Sales rolling stats
        for agg_func in ['mean', 'std', 'min', 'max']:
            col_name = f'{target}_roll_{window}_{agg_func}'
            df_features[col_name] = df_features.groupby('store')[target].transform(
                lambda x: x.rolling(window=window, min_periods=1).agg(agg_func)
            )
            time_varying_unknown_reals.append(col_name)
        
        # Transaction count rolling stats
        col_name = f'transaction_count_roll_{window}_mean'
        df_features[col_name] = df_features.groupby('store')['transaction_count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        time_varying_unknown_reals.append(col_name)
    
    # Create TFT special features dictionary
    tft_features = {
        'static_categoricals': static_categoricals,
        'static_reals': static_reals,
        'time_varying_known_categoricals': time_varying_known_categoricals,
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_categoricals': time_varying_unknown_categoricals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'target': target
    }
    
    # Add feature groups as attributes to dataframe
    for k, v in tft_features.items():
        setattr(df_features, k, v)
    
    # Fill NaN values with appropriate values for each column type
    for col in df_features.select_dtypes(include=['float64']).columns:
        df_features[col] = df_features[col].fillna(0)
    
    for col in df_features.select_dtypes(include=['object', 'category']).columns:
        df_features[col] = df_features[col].fillna('missing')
    
    return df_features

def prepare_tft_dataset(df: pd.DataFrame, target: str = 'sale_dollars', 
                       max_prediction_length: int = 30,
                       max_encoder_length: int = 90) -> dict:
    """
    Prepare dataset for TFT model.
    
    Args:
        df (pd.DataFrame): Input dataframe with features
        target (str): Target variable
        max_prediction_length (int): Maximum prediction length
        max_encoder_length (int): Maximum encoder length
        
    Returns:
        dict: Dictionary with feature groups for TFT
    """
    # Define feature groups
    static_categoricals = ['store', 'city', 'county', 'zipcode']
    static_reals = ['lon', 'lat']
    
    time_varying_known_categoricals = ['is_weekend', 'is_holiday']
    time_varying_known_reals = [
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'days_to_nearest_holiday'
    ]
    
    # Get time-varying unknown features
    time_varying_unknown_reals = [
        col for col in df.columns 
        if col.startswith(('lag_', 'rolling_')) and col.endswith(f'_{target}')
    ]
    
    # Add target to unknown reals
    time_varying_unknown_reals.append(target)
    
    return {
        'static_categoricals': static_categoricals,
        'static_reals': static_reals,
        'time_varying_known_categoricals': time_varying_known_categoricals,
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'max_prediction_length': max_prediction_length,
        'max_encoder_length': max_encoder_length
    } 