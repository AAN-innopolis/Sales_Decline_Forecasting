"""
Base feature engineering module for common operations.
Contains functions for data ingestion, validation, and basic feature creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import statistics
import logging
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import setup_logger


def validate_and_clean_data(
        df: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Validate and clean input data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger: Logger instance
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """ 
    logger.info("Validating input data")
    try:
        duplicates = df[df['invoice_line_no'].duplicated()]
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} duplicate transactions")
        df = df.drop_duplicates(subset=['invoice_line_no'])

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['store', 'date'])
    except Exception as e:
        raise Exception(f"Error while validating data: {e}")
    
    return df

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

def safe_median(arr):
    return np.median(arr) if len(arr) > 0 else 0

def expand_by_weight(values, weights):
    values = np.asarray(values)
    weights = np.abs(np.asarray(weights)).astype(int)
    if len(values) != len(weights):
        return values
    return np.repeat(values, weights)

def get_base_statistics(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create base statistics
    Only includes essential aggregations per store and date.
    Requires the data of transaction level information.
    Returns the features which need to be added to the dataframe by key [store, date]
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger: Logger instance
        
    Returns:
        pd.DataFrame: Aggregated dataframe per store and date containing base statistics
    """
    df = df_original.copy()
    logger.info("Creating base statistics")

    df['state_bottle_cost_total'] = df['state_bottle_cost'] * df['sale_bottles']
    df['bottle_volume_ml_total'] = df['bottle_volume_ml'] * df['sale_bottles']
    df['pack_number'] = np.ceil(df['sale_bottles'] / df['pack'])
    df['pack_volume'] = df['pack_number'] * df['pack']
    
    statistics = (
        df.groupby(['store', 'date']).agg({
                'sale_bottles': 'sum',
                'sale_dollars': 'sum',
                'sale_liters': 'sum',
                'pack_number': 'sum',
                'state_bottle_cost_total': 'sum', # state_bottle_cost - amount that ABD paid for all bottles
                'pack_volume': 'sum',
                'bottle_volume_ml_total': 'sum',
                'invoice_line_no': 'count',
                'category': 'nunique',
                'itemno': 'nunique'
            })
            .reset_index()
    )
    print(statistics.columns)

    statistics['state_bottle_cost_avg'] = (
         statistics['state_bottle_cost_total'] / statistics['sale_bottles']
    )
    statistics['bottle_volume_ml_avg'] = (
         statistics['bottle_volume_ml_total'] / statistics['sale_bottles']
    )
    statistics['pack_avg'] = (
         statistics['pack_volume'] / statistics['pack_number']
    )

    statistics.drop(columns=['state_bottle_cost_total', 'pack_volume', 'pack_number',
                             'bottle_volume_ml_total'], inplace=True)
            
    statistics.rename(columns={
        'sale_bottles': 'purchased_bottles',
        'sale_dollars': 'purchase_amount',
        'sale_liters': 'purchased_liters',
        'state_bottle_cost_avg': 'average_state_bottle_cost',
        'pack_avg': 'average_pack',
        'bottle_volume_ml_avg': 'average_bottle_volume',
        'invoice_line_no': 'transaction_count',
        'category': 'unique_categories',
        'itemno': 'unique_items'
    })

    logger.info(f"Base statistics created. 
                Shape: {statistics.shape}, 
                Columns: {statistics.columns}")
    return statistics


def get_extended_statistics(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create additional statistics.
    Requires the data of transaction level information.
    Returns the features which need to be added to the dataframe by key [store, date]
    
    Args:
        df_original: pd.DataFrame, 
        logger: logging.Logger
        
    Returns:
        pd.DataFrame: Aggregated dataframe per store and date containing additional statistics
    """
    df = df_original.copy()
    logger.info("Creating additional statistics")

    df['pack_number'] = np.ceil(df['sale_bottles'] / df['pack'])

    statistics = df.groupby(['store', 'date']).agg({
        # Sales - basic metrics and distributions
        'sale_bottles': ['mean', 'median', 'min', 'max'],
        'sale_dollars': ['mean', 'median', 'min', 'max'],
        'sale_liters': ['mean', 'median', 'min', 'max'],
        # Statistics on bottle costs
        'sale_bottles': list,
        'state_bottle_cost': list,
        # Statistics on packs and volumes
        'pack': list,
        'pack_number': list,
        'bottle_volume_ml': list
    }).reset_index()
    statistics.columns = ['_'.join(col).strip('_') for col in statistics.columns.values]
    statistics['expanded_costs'] = statistics.apply(
		lambda row: expand_by_weight(row['state_bottle_cost_list'], row['sale_bottles_list']),
		axis=1
	)
    statistics['expanded_volumes'] = statistics.apply(
		lambda row: expand_by_weight(row['bottle_volume_ml_list'], row['sale_bottles_list']),
		axis=1
	)
    statistics['expanded_packs'] = statistics.apply(
		lambda row: expand_by_weight(row['pack_list'], row['pack_number_list']),
		axis=1
	)

    statistics['state_bottle_cost_mean'] = statistics['expanded_costs'].apply(safe_mean)
    statistics['state_bottle_cost_median'] = statistics['expanded_costs'].apply(safe_median)
    statistics['bottle_volume_ml_mean'] = statistics['expanded_volumes'].apply(safe_mean)
    statistics['bottle_volume_ml_median'] = statistics['expanded_volumes'].apply(safe_median)
    statistics['pack_mean'] = statistics['expanded_packs'].apply(safe_mean)
    statistics['pack_median'] = statistics['expanded_packs'].apply(safe_median)
    statistics.drop(columns=['expanded_costs', 'expanded_volumes', 'expanded_packs'], inplace=True)

    statistics['state_bottle_cost_min'] = statistics['state_bottle_cost_list'].apply(min)
    statistics['pack_min'] = statistics['pack_list'].apply(min)
    statistics['bottle_volume_ml_min'] = statistics['bottle_volume_ml_list'].apply(min)
    statistics['state_bottle_cost_max'] = statistics['state_bottle_cost_list'].apply(max)
    statistics['pack_max'] = statistics['pack_list'].apply(max)
    statistics['bottle_volume_ml_max'] = statistics['bottle_volume_ml_list'].apply(max)
    statistics.drop(columns=['state_bottle_cost_list', 'bottle_volume_ml_list',
                             'pack_list', 'pack_number_list', 'sale_bottles_list'], inplace=True)
    logger.info(f"Extended statistics created. 
                Shape: {statistics.shape}, 
                Columns: {statistics.columns}")
    return statistics


def get_derived_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create derived features.
    Requires the data of date level information.
    Returns the features which need to be added to the dataframe by key [store, date]

    Args:
        df (pd.DataFrame): Input dataframe
        logger: Logger instance
        
    Returns:
        pd.DataFrame: Derived features
    """
    df = df_original.copy()
    logger.info("Creating derived features")

    df['avg_price_per_bottle'] = df['sale_dollars'] / df['sale_bottles'].replace(0, np.nan)
    df['avg_price_per_liter'] = df['sale_dollars'] / df['sale_liters'].replace(0, np.nan)
    df['avg_items_per_transaction'] = df['unique_items'] / df['transaction_count'].replace(0, np.nan)
    df['avg_transaction_value'] = df['sale_dollars'] / df['transaction_count'].replace(0, np.nan)

    df_derived = df[[   
        'avg_price_per_bottle', 
        'avg_price_per_liter', 
        'avg_items_per_transaction', 
        'avg_transaction_value'
    ]]
    logger.info(f"Derived features created. 
                Shape: {df_derived.shape}, 
                Columns: {df_derived.columns}")
    return df_derived


def get_store_attributes(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create static store attributes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Store attributes
    """
    df = df_original.copy()
    logger.info("Creating store attributes")

    store_attributes = df.groupby('store').agg({
        'name': lambda x: x.mode()[0],
        'address': lambda x: x.mode()[0],
        'city': lambda x: x.mode()[0],
        'zipcode': lambda x: x.mode()[0],
        'store_location': lambda x: x.mode()[0],
        'county': lambda x: x.mode()[0]
    }).reset_index()
    
    store_locations = store_attributes['store_location'].apply(eval)
    store_attributes['lon'] = store_locations.apply(lambda x: x['coordinates'][0])
    store_attributes['lat'] = store_locations.apply(lambda x: x['coordinates'][1])
    
    logger.info(f"Store attributes created. 
                Shape: {store_attributes.shape}, 
                Columns: {store_attributes.columns}")
    return store_attributes


def get_holiday_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        country_code: str = 'US', 
        holiday_diff_threshold: int = 0
    ) -> pd.DataFrame:
    """
    Get holiday features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        country_code (str): Country code for holidays
        holiday_diff_threshold (int): Threshold for holiday proximity
        logger: Logger instance
        
    Returns:
        pd.DataFrame: DataFrame with holiday features
    """
    df = df_original.copy()
    logger.info("Creating holiday features")
        
    try:
        import holidays
        unique_years = df['year'].unique()
        all_holidays = pd.concat([
            pd.DataFrame({
                'date': list(holidays.country_holidays(country_code, years=year).keys()),
                'name': list(holidays.country_holidays(country_code, years=year).values())
            }) for year in unique_years
        ])
        all_holidays['date'] = pd.to_datetime(all_holidays['date'])
        
        dates = df.index.unique()
        holiday_features = pd.DataFrame(index=dates)
        
        time_diffs = (all_holidays['date'].values[:, None] - dates.values).astype('timedelta64[D]').astype(int)
        nearest_idx = np.argmin(np.abs(time_diffs), axis=0)
        days_diff = time_diffs[nearest_idx, np.arange(len(dates))]
        
        mask = np.abs(days_diff) <= holiday_diff_threshold
        holiday_features['is_holiday'] = mask.astype(int)
        holiday_features['holiday_name'] = np.where(mask, all_holidays.iloc[nearest_idx]['name'].values, '')
        holiday_features['days_to_nearest_holiday'] = days_diff
        
        return holiday_features
    except Exception as e:
        warnings.warn(f"Failed to add holiday features: {e}")
        return pd.DataFrame(index=df.index)


def create_temporal_features(df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Create temporal features that are common for all models.
    Only includes basic cyclical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger: Logger instance
        
    Returns:
        pd.DataFrame: Dataframe with temporal features
    """
    if logger is None:
        logger = setup_logger()
        
    logger.info("Creating temporal features")
    
    result_df = df.copy()
    
    # Set date as index for temporal features
    result_df = result_df.set_index('date')
    
    # Cyclical features
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
    
    result_df['month'] = result_df.index.month
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    
    result_df['quarter'] = result_df.index.quarter
    result_df['quarter_sin'] = np.sin(2 * np.pi * result_df['quarter'] / 4)
    result_df['quarter_cos'] = np.cos(2 * np.pi * result_df['quarter'] / 4)
    
    # Year and weekend flag
    result_df['year'] = result_df.index.year
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Drop intermediate columns
    result_df = result_df.drop(columns=['day_of_week', 'month', 'quarter'])
    
    return result_df.reset_index() 