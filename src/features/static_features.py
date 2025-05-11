"""
Static Features Module

This module creates features that don't require historical data or complex aggregations.
It includes:
1. Static store attributes (location, address, etc.)
2. Holiday and calendar features
3. Time-based cyclical features

These features can be created independently of historical data and are used
to provide context and temporal information for the analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.configs import settings


def get_store_attributes(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Creates static store attributes that don't change over time:
    - Store name and address
    - City and county information
    - Geographic coordinates
    - ZIP code
    
    These attributes provide location and identification information for each store.
    
    Args:
        df_original (pd.DataFrame): Transaction or aggregated data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Static attributes for each store
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
    
    # Extract longitude and latitude from store_location
    store_attributes['lon'], store_attributes['lat'] = 0, 0
    store_locations = store_attributes.loc[
        store_attributes['store_location'] != 'Unknown', 
        'store_location'].apply(eval)
    store_attributes.loc[
        store_attributes['store_location'] != 'Unknown', 'lon'
        ] = store_locations.apply(lambda x: x['coordinates'][0])
    store_attributes.loc[
        store_attributes['store_location'] != 'Unknown', 'lat'
        ] = store_locations.apply(lambda x: x['coordinates'][1])
    store_attributes.drop(columns=['store_location'], inplace=True)

    logger.info(f"Store attributes created.\
                Shape: {store_attributes.shape},\
                Columns: {store_attributes.columns}")
    return store_attributes

def get_holiday_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        country_code: str = 'US',
        holiday_diff_threshold: int = 0
    ) -> pd.DataFrame:
    """
    Creates features related to holidays and special dates:
    - Holiday indicators
    - Days until/since nearest holiday
    - Holiday proximity flags
    
    These features help capture seasonal patterns and special event impacts.
    
    Args:
        df_original (pd.DataFrame): Transaction or aggregated data
        logger (logging.Logger): Logger instance
        country_code (str): Country code for holiday calendar
        holiday_diff_threshold (int): Days threshold for holiday proximity
        
    Returns:
        pd.DataFrame: Holiday-related features for each date
    """
    df = df_original.copy()
    logger.info("Creating holiday features")
        
    try:
        import holidays
        unique_years = df['date'].dt.year.unique()
        all_holidays = pd.concat([
            pd.DataFrame({
                'date': list(
                    holidays.country_holidays(
                        country_code, 
                        years=year).keys()),
                'name': list(
                    holidays.country_holidays(
                        country_code, 
                        years=year).values())
            }) for year in unique_years
        ])
        all_holidays['date'] = pd.to_datetime(all_holidays['date'])
        dates = df.set_index('date').index.unique()
        time_diffs = (
            all_holidays['date'].values[:, None] 
            - dates.values
        ).astype('timedelta64[D]').astype(int)
        nearest_idx = np.argmin(np.abs(time_diffs), axis=0)
        days_diff = time_diffs[nearest_idx, np.arange(len(dates))]
        mask = np.abs(days_diff) <= holiday_diff_threshold

        holiday_features = pd.DataFrame(index=dates)
        holiday_features['is_holiday'] = mask.astype(int)
        holiday_features['holiday_name'] = np.where(
            mask, 
            all_holidays.iloc[nearest_idx]['name'].values, 
            'unknown'
        )
        holiday_features['days_to_nearest_holiday'] = days_diff
        holiday_features = holiday_features.reset_index()
        
        logger.info(f"Holiday features created.\
                    Shape: {holiday_features.shape},\
                    Columns: {holiday_features.columns}")
        return holiday_features
    except Exception as e:
        raise Exception(f"Error while creating holiday features: {e}")

def get_cyclical_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        llm: bool = False
    ) -> pd.DataFrame:
    """
    Creates cyclical time-based features using sine and cosine transformations:
    - Day of week (cyclic)
    - Day of month (cyclic)
    - Month (cyclic)
    - Quarter (cyclic)
    - Week of year (cyclic)
    
    These features help capture seasonal patterns and temporal cycles.
    
    Args:
        df_original (pd.DataFrame): Transaction or aggregated data
        logger (logging.Logger): Logger instance
        llm (bool): Whether to use LLM-friendly feature format
        
    Returns:
        pd.DataFrame: Cyclical time features for each date
    """
    df = df_original.groupby(['date']).agg({
        'date': 'first'
    }).reset_index(drop=True)
    logger.info("Creating cyclical features")

    # Day of week (0-6)
    df['day_of_week'] = df['date'].dt.dayofweek 
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    # Day of month (1-28|29|30|31)
    df['day_of_month'] = df['date'].dt.day  
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    # Month (1-12)
    df['month'] = df['date'].dt.month  
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Quarter (1-4)
    df['quarter'] = df['date'].dt.quarter  
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    # Week of year (1-52)
    df['week_of_year'] = df['date'].dt.isocalendar().week  
    df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    # Year
    df['year'] = df['date'].dt.year
    
    if llm:
        df['weekday'] = df['date'].dt.weekday
        df['month_name'] = df['date'].dt.month_name()
        df = df[[
            'date',
            'weekday',
            'month_name'
        ]]
    else:
        df = df[[
            'date',
            'day_of_week_sin',
            'day_of_week_cos',
            'day_of_month_sin',
            'day_of_month_cos',
            'month_sin',
            'month_cos',
            'quarter_sin',
            'quarter_cos',
            'week_of_year_sin',
            'week_of_year_cos',
            'year',
        ]]
    logger.info(f"Cyclical features created.\
                Shape: {df.shape},\
                Columns: {df.columns}")
    return df
