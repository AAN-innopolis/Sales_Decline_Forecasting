"""
Base feature engineering module for common operations.
Contains functions for data ingestion, validation, and basic feature creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import setup_logger
from src.config.configs import settings

def validate_and_clean_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        min_stores_count: int = 5,
        max_days_between_purchases: int = 200 # max = 1813
    ) -> pd.DataFrame:
    """
    Validate and clean input data.
    
    Args:
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """ 
    logger.info("Validating input data")
    df = df_original.copy()
    logger.info(f"Initial shape: {df.shape}")
    try:
        duplicates = df[df['invoice_line_no'].duplicated()]
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} duplicate transactions")
        df = df.drop_duplicates(subset=['invoice_line_no'])
        logger.info(f"After dropping duplicates: {df.shape}")

        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Start date: {df['date'].min()}, End date: {df['date'].max()}")
        df = df.sort_values(['store', 'date'])

        store_counts = df.groupby(['store']).size().sort_values(ascending=False)
        rare_stores_by_count = list(store_counts[store_counts<=min_stores_count].index)
        logger.info(f"Filtered out {len(rare_stores_by_count)} stores with less than {min_stores_count} transactions: {rare_stores_by_count}")

        df['diff'] = (
            df.drop_duplicates(subset=['store', 'date'])
            .groupby('store')['date']
            .diff()
            .dt.days
            .replace(np.nan, -1)
        )
        diff_counts = df['diff'].value_counts()
        rare_diffs = diff_counts[diff_counts.index > max_days_between_purchases].index.values
        rare_stores_by_diff = list(df[df['diff'].isin(rare_diffs)]['store'].unique())
        logger.info(f"Filtered out {len(rare_stores_by_diff)} stores with more than {max_days_between_purchases} days between transactions: {rare_stores_by_diff}")
        df = df[~df['store'].isin(rare_stores_by_diff + rare_stores_by_count)]
        logger.info(f"After filtering: {df.shape}")

    except Exception as e:
        raise Exception(f"Error while validating data: {e}")
    
    return df


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
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Aggregated dataframe per store and date containing base statistics
    """
    df = df_original.copy()
    logger.info("Creating base statistics")
    
    statistics = (
        df.groupby(['store', 'date']).agg({
                'sale_bottles': 'sum',
                'sale_dollars': 'sum',
                'sale_liters': 'sum',
                'state_bottle_cost': 'sum', # state_bottle_cost - amount that ABD paid for each bottle
                'state_bottle_retail': 'sum', # amount that the store paid for each bottle
                'pack': 'sum',
                'bottle_volume_ml': 'sum',
                'invoice_line_no': 'count',
                'category': 'nunique',
                'itemno': 'nunique'
            })
            .reset_index()
            .rename(columns={
                'sale_bottles': 'purchased_bottles',
                'sale_dollars': 'purchase_amount',
                'sale_liters': 'purchased_liters',
                'state_bottle_cost': 'total_state_bottle_cost',
                'state_bottle_retail': 'total_state_bottle_retail',
                'pack': 'total_packs',
                'bottle_volume_ml': 'total_bottle_volume',
                'invoice_line_no': 'transaction_count',
                'category': 'unique_categories',
                'itemno': 'unique_items'
            })
    )
    logger.info(f"Base statistics created.\
                Shape: {statistics.shape}, \
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
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Aggregated dataframe per store and date containing additional statistics
    """
    df = df_original.copy()
    logger.info("Creating additional statistics")

    statistics = df.groupby(['store', 'date']).agg({
        # Sales - basic metrics and distributions
        'sale_bottles': ['mean', 'median', 'min', 'max'],
        'sale_dollars': ['mean', 'median', 'min', 'max'],
        'sale_liters': ['mean', 'median', 'min', 'max'],
        # Statistics on bottle costs
        'state_bottle_cost': ['mean', 'median', 'min', 'max'],
        'state_bottle_retail': ['mean', 'median', 'min', 'max'],
        # Statistics on packs and volumes
        'pack': ['mean', 'median', 'min', 'max'],
        'bottle_volume_ml': ['mean', 'median', 'min', 'max']
    }).reset_index()
    statistics.columns = ['_'.join(col).strip('_') for col in statistics.columns.values]
    logger.info(f"Extended statistics created.\
                Shape: {statistics.shape},\
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
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Derived features
    """
    df = df_original.copy()
    logger.info("Creating derived features")

    df['days_since_prev_purchase'] = df.groupby('store')['date'].diff().dt.days
    df['days_since_prev_purchase'] = df['days_since_prev_purchase'].fillna(-1)
    ## Key metrics for analyzing store purchases from vendor by store and purchase date
    # Average purchase price per bottle - helps analyze purchase efficiency and track price trends
    df['avg_price_per_bottle'] = df['purchase_amount'] / df['purchased_bottles'].replace(0, np.nan)
    # Average purchase price per liter - enables comparison of purchase prices across different volume products
    df['avg_price_per_liter'] = df['purchase_amount'] / df['purchased_liters'].replace(0, np.nan)
    # Average number of unique items per purchase - indicates assortment diversity
    df['avg_items_per_transaction'] = df['unique_items'] / df['transaction_count'].replace(0, np.nan)
    # Average purchase value - important indicator of purchase volume
    df['avg_transaction_value'] = df['purchase_amount'] / df['transaction_count'].replace(0, np.nan)
    # Purchase margin - shows difference between recommended retail price and purchase price
    df['profit_margin'] = (df['total_state_bottle_retail'] - df['total_state_bottle_cost']) / df['total_state_bottle_retail'].replace(0, np.nan)
    # Markup factor - helps analyze purchase conditions and track changes in pricing policy
    df['discount_factor'] = df['total_state_bottle_retail'] / df['total_state_bottle_cost'].replace(0, np.nan)

    df_derived = df[[ 
        'store', 
        'date',
        'days_since_prev_purchase',
        'avg_price_per_bottle', 
        'avg_price_per_liter', 
        'avg_items_per_transaction', 
        'avg_transaction_value',
        'profit_margin',
        'discount_factor'
    ]]
    logger.info(f"Derived features created.\
                Shape: {df_derived.shape},\
                Columns: {df_derived.columns}")
    return df_derived


def get_store_attributes(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create static store attributes.
    
    Args:
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        
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
    
    store_locations = store_attributes['store_location'].apply(lambda x: eval(x) if pd.notna(x) else None)
    store_attributes['lon'] = store_locations.apply(lambda x: x['coordinates'][0] if pd.notna(x) else None)
    store_attributes['lat'] = store_locations.apply(lambda x: x['coordinates'][1] if pd.notna(x) else None)
    
    logger.info(f"Store attributes created.\
                Shape: {store_attributes.shape},\
                Columns: {store_attributes.columns}")
    return store_attributes


def get_item_details(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create item details.
    Requires the data of transaction level information.
    Returns the features which need to be added to the dataframe by key [store, date]

    Args:
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance

    Returns:
        pd.DataFrame: DataFrame with item details
    """
    df = df_original.copy()
    logger.info("Getting item details")
    item_details = df.groupby(['store', 'date']).apply(
        lambda x: x[settings.ITEM_DETAILS_COLUMNS]
        .drop_duplicates()
        .to_dict('records'), include_groups=False
    ).reset_index(name='item_details')
    logger.info(f"Item details created.\
                Shape: {item_details.shape},\
                Columns: {item_details.columns}")
    return item_details



def get_holiday_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        country_code: str = 'US', 
        holiday_diff_threshold: int = 0
    ) -> pd.DataFrame:
    """
    Get holiday features.
    
    Args:
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        country_code (str): Country code for holidays
        holiday_diff_threshold (int): Threshold for holiday proximity
        
    Returns:
        pd.DataFrame: Holiday features
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
            '')
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
    Create cyclical features.
    
    Args:
        df_original (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
        llm (bool): Whether to create cyclical features for LLM
    Returns:
        pd.DataFrame: Cyclical features
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


def get_store_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create features related to stores and locations.
    Requires the data of date level information.
    Returns the features which need to be added to the dataframe by key [store, date]
    
    Args:
        df_original (pandas.DataFrame): Dataframe with data
        logger (logging.Logger): Logger instance
        
    Returns:
        pandas.DataFrame: Store features
    """
    logger.info("Creating features related to stores")
    df = df_original.copy()
   
    q33, q66 = (
        df.groupby('date')['purchase_amount']
        .median()
        .expanding()
        .quantile(0.33)
        .shift(1)
    ).rename('q33'), (
        df.groupby('date')['purchase_amount']
        .median()
        .expanding()
        .quantile(0.66)
        .shift(1)
    ).rename('q66')
    global_sales_quantiles = pd.concat([q33, q66], axis=1)
    city_means = (
        df.groupby(['city','date'])['purchase_amount'].median()
        .expanding()
        .mean()
        .shift(1)
    )
    county_means = (
        df.groupby(['county', 'date'])['purchase_amount'].median()
        .expanding()
        .mean()
        .shift(1)
    )

    store_dfs = []
    for _, store in df.groupby('store'):
        store_df = pd.DataFrame( 
            {'store': store['store'].values},
            index=store['date']
        )
        store_df['store_avg_sales'] = (
            store['purchase_amount']
            .expanding(min_periods=1)
            .median()
            .shift(1)
        )
        store_df['store_avg_transactions'] = (
            store['transaction_count']
            .expanding(min_periods=1)
            .mean()
            .shift(1)
        )
        store_df['store_avg_items'] = (
            store['unique_items']
            .expanding(min_periods=1)
            .mean()
            .shift(1)
        )
        # Get historical quantiles for store classification
        # Get quantiles for the current store's dates
        current_quantiles = global_sales_quantiles.loc[store_df.index]
        store_df['store_size'] = 1  # Small store by default
        
        # Classify store size based on historical performance
        store_df.loc[store_df['store_avg_sales'] > current_quantiles['q66'], 'store_size'] = 3  # Large store
        store_df.loc[store_df['store_avg_sales'] > current_quantiles['q33'], 'store_size'] = 2  # Medium store
        
        # Get city and county historical averages
        current_city = store['city'].iloc[0]
        current_county = store['county'].iloc[0]
        store_df['city_avg_sales'] = city_means.loc[current_city].loc[store_df.index]
        store_df['county_avg_sales'] = county_means.loc[current_county].loc[store_df.index]
        
        # Calculate ratios using historical averages
        store_df['store_to_city_sales_ratio'] = (
            store_df['store_avg_sales'] / 
            store_df['city_avg_sales'].replace(0, np.nan)
        )
        store_df['store_to_county_sales_ratio'] = (
            store_df['store_avg_sales'] / 
            store_df['county_avg_sales'].replace(0, np.nan)
        )
        store_dfs.append(store_df.set_index('store', append=True))
        
    store_dfs = pd.concat(store_dfs).reset_index()
    logger.info(f"Store features created. \
                Shape: {store_dfs.shape}, \
                Columns: {store_dfs.columns}")
    return store_dfs

