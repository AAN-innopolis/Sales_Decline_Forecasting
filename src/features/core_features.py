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
        logger: logging.Logger
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
    # print(statistics.columns)

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
    logger.info(f"Base statistics created.\
                Shape: {statistics.shape},\
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

    df_derived = df[[ 
        'store', 
        'date',
        'days_since_prev_purchase',
        'avg_price_per_bottle', 
        'avg_price_per_liter', 
        'avg_items_per_transaction', 
        'avg_transaction_value'
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
    
    store_locations = store_attributes['store_location'].apply(eval)
    store_attributes['lon'] = store_locations.apply(lambda x: x['coordinates'][0])
    store_attributes['lat'] = store_locations.apply(lambda x: x['coordinates'][1])
    
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

