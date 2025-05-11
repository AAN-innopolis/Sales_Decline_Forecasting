"""
Store-Date Level Features Module

This module creates features that require store-date level data and historical information.
It performs the following operations:
1. Creates derived features that depend on previous purchases
2. Generates store-specific features using historical performance
3. Calculates store rankings and comparisons with city/county averages

The features in this module require data to be pre-aggregated to store-date level
and are used for analyzing store performance and purchase patterns over time.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.configs import settings


def get_derived_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Creates features derived from store-date level data, including:
    - Days since previous purchase
    - Average price per bottle and liter
    - Average items per transaction
    - Average transaction value
    
    These features help analyze purchase patterns and efficiency metrics.
    
    Args:
        df_original (pd.DataFrame): Store-date level aggregated data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Derived features for each store-date combination
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
                \rColumns: \n{df_derived.columns}")
    return df_derived

def get_store_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Creates store-specific features using historical performance data:
    - Store size classification (small/medium/large)
    - Historical average sales and transactions
    - Comparison with city and county averages
    - Store performance ratios
    
    These features help understand store characteristics and relative performance.
    
    Args:
        df_original (pd.DataFrame): Store-date level aggregated data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Store-specific features for each store-date combination
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
    logger.info(f"Store features created. \n\
                \rShape: {store_dfs.shape}, \n\
                \rColumns: \n{store_dfs.columns}")
    return store_dfs
