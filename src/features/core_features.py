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
        #### Dropping rows with missing values
        notna_cols = settings.NOTNA_COLUMNS
        logger.info(f"Checking columns for appearance of missing values: {notna_cols}")
        for col in notna_cols:
            if df[col].isna().any():
                logger.info(f"{col}: missing values: {df[df[col].isna()].shape[0]} rows. Dropping rows with missing values...")
                df = df.dropna(subset=[col])
                logger.info(f"After dropping: {df.shape[0]} rows")

        #### Dropping duplicates by invoice_line_no
        logger.info(f"Checking duplicates by {settings.PRIMARY_KEYS}")
        duplicates = df[df[settings.PRIMARY_KEYS].duplicated()]
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} duplicate transactions")
        df = df.drop_duplicates(subset=settings.PRIMARY_KEYS)
        logger.info(f"After dropping duplicates: {df.shape[0]} rows")

        #### Dropping rows with zero sale_bottles
        logger.info(f"Checking {settings.NONZERO_COLUMNS} with appearance of zero values")
        for col in settings.NONZERO_COLUMNS:
            if (df[col] == 0).any():
                logger.info(f"{col}: zero values: {df[df[col] == 0].shape[0]} rows. Dropping rows with zero values...")
                df = df[df[col] != 0]
                logger.info(f"After dropping zero values: {df.shape[0]} rows")

        #### Filling missing values with -1
        logger.info(f"Checking columns with missing values.")
        for col in settings.INDEXED_COLUMNS:
            if df[col].isna().any():
                logger.info(f"{col}: missing values: {df[df[col].isna()].shape[0]} rows. Filling with -1...")
                df.loc[df[col].isna(), col] = -1

        for col in settings.CATEGORICAL_COLUMNS:
            if df[col].isna().any():
                logger.info(f"{col}: missing values: {df[df[col].isna()].shape[0]} rows. Filling with 'Unknown'...")
                df.loc[df[col].isna(), col] = 'Unknown'

        #### Droppping stores with less than min_stores_count transactions 
        #### and stores with more than max_days_between_purchases days between transactions
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
        logger.info(f"After filtering: {df.shape[0]} rows")

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

    statistics['state_bottle_cost_avg'] = (
         statistics['state_bottle_cost_total'] / statistics['sale_bottles'].replace(0, np.nan)
    ).replace(np.nan, 0)
    statistics['bottle_volume_ml_avg'] = (
         statistics['bottle_volume_ml_total'] / statistics['sale_bottles'].replace(0, np.nan)
    ).replace(np.nan, 0)
    statistics['pack_avg'] = (
         statistics['pack_volume'] / statistics['pack_number'].replace(0, np.nan)
    ).replace(np.nan, 0)

    statistics = statistics.drop(columns=['state_bottle_cost_total', 'pack_volume', 'pack_number',
                             'bottle_volume_ml_total'])
            
    statistics = statistics.rename(columns={
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
    logger.info(f"Base statistics created.\n\
                \rShape: {statistics.shape},\n\
                \rColumns: {statistics.columns}")
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
    
    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Vectorized O(n log n) weighted median."""
        sorter = np.argsort(values)
        v_sorted = values[sorter]
        w_sorted = weights[sorter]
        cum_w = w_sorted.cumsum()
        cut = w_sorted.sum() / 2
        return v_sorted[cum_w >= cut][0]

    df = df_original.copy()
    logger.info("Creating additional statistics")

    # number of packs actually delivered
    df['pack_number'] = (np.ceil(df['sale_bottles'] / df['pack'].replace(0, np.nan))).replace(np.nan, 0)

    # pre-multiply one time
    df['cost_full'] = df['state_bottle_cost'] * df['sale_bottles']
    df['vol_full'] = df['bottle_volume_ml'] * df['sale_bottles']
    df['pack_full'] = df['pack'] * df['pack_number']

    grp_cols = ['store', 'date']
    g = df.groupby(grp_cols, sort=False)

    basic = g.agg({
        'sale_bottles'     : ['mean', 'median', 'min', 'max', 'sum'],
        'sale_dollars'     : ['mean', 'median', 'min', 'max'],
        'sale_liters'      : ['mean', 'median', 'min', 'max'],
        'state_bottle_cost': ['min', 'max'],
        'bottle_volume_ml' : ['min', 'max'],
        'pack'             : ['min', 'max'],
        'pack_number'      : ['sum']
    })

    basic.columns = [
        f"{col}_{func}" if func else col
        for col, func in basic.columns.to_flat_index()
    ]

    totals = g.agg(
        cost_total = ('cost_full', 'sum'),
        vol_total  = ('vol_full',  'sum'),
        pack_total = ('pack_full', 'sum')
    )

    means = pd.DataFrame({
        'state_bottle_cost_mean': (totals['cost_total'] / basic['sale_bottles_sum'].replace(0, np.nan)).replace(np.nan, 0),
        'bottle_volume_ml_mean': (totals['vol_total']  / basic['sale_bottles_sum'].replace(0, np.nan)).replace(np.nan, 0),
        'pack_mean': (totals['pack_total'] / basic['pack_number_sum'].replace(0, np.nan)).replace(np.nan, 0)
    })

    def _medians(sub: pd.DataFrame) -> pd.Series:
        sb = sub['sale_bottles'].abs().to_numpy() # weights for cost & volume
        pn = sub['pack_number'].abs().to_numpy() # weights for pack
        return pd.Series({
            'state_bottle_cost_median': weighted_median(sub['state_bottle_cost'].to_numpy(), sb),
            'bottle_volume_ml_median': weighted_median(sub['bottle_volume_ml'].to_numpy(), sb),
            'pack_median': weighted_median(sub['pack'].to_numpy(), pn)
        })
    
    medians = g.apply(_medians, include_groups=False)

    stats = (
        basic
        .drop(columns=['sale_bottles_sum', 'pack_number_sum'])
        .join(means)
        .join(medians)
        .reset_index()
        .astype({'store': df['store'].dtype})
    )
    logger.info(f"Extended statistics created.\n\
                \rShape: {stats.shape},\n\
                \rColumns: {stats.columns}")
    return stats

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
    logger.info(f"Derived features created.\n\
                \rShape: {df_derived.shape},\n\
                \rColumns: {df_derived.columns}")
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
    
    store_attributes[['lon', 'lat']] = None
    mask = store_attributes['store_location'].notna()
    coords = (
        store_attributes.loc[mask, 'store_location']
        .apply(lambda x: eval(x)['coordinates'])
    )

    store_attributes.loc[mask, ['lon', 'lat']] = pd.DataFrame(coords.tolist(), index=coords.index)

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
    logger.info(f"Item details created.\n\
                \rShape: {item_details.shape},\n\
                \rColumns: {item_details.columns}")
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
        
        logger.info(f"Holiday features created.\n\
                    \rShape: {holiday_features.shape},\n\
                    \rColumns: {holiday_features.columns}")
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
    logger.info(f"Cyclical features created.\n\
                \rShape: {df.shape},\n\
                \rColumns: {df.columns}")
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
    logger.info(f"Store features created.\n\
                \rShape: {store_dfs.shape},\n\
                \rColumns: {store_dfs.columns}")
    return store_dfs

