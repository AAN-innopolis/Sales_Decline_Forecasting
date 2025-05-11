"""
Transaction Aggregation Features Module

This module handles the first level of feature engineering by aggregating raw transaction data
into store-date level features. It performs the following operations:
1. Validates and cleans the input transaction data
2. Creates base statistics by aggregating transactions per store and date
3. Generates extended statistics with more detailed metrics
4. Extracts item-level details for each store-date combination

The output of this module serves as the foundation for further feature engineering steps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.configs import settings


def validate_and_clean_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        min_stores_count: int = 5,
        max_days_between_purchases: int = 200 # max = 1813
    ) -> pd.DataFrame:
    """
    Validates and cleans the input transaction data by:
    1. Removing stores with too few transactions
    2. Handling missing values in categorical and indexed columns
    3. Removing rows with zero values in critical columns
    4. Filtering out stores with irregular purchase patterns

    Args:
        df_original (pd.DataFrame): Raw transaction data
        logger (logging.Logger): Logger instance for tracking operations
        min_stores_count (int): Minimum number of transactions required per store
        max_days_between_purchases (int): Maximum allowed days between purchases
        
    Returns:
        pd.DataFrame: Cleaned transaction data ready for aggregation
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
        df = df.drop(columns=['diff'])
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
    Creates essential store-date level statistics by aggregating transaction data.
    These statistics include:
    - Total bottles, dollars, and liters purchased
    - Average costs and volumes
    - Transaction counts and unique items
    
    Args:
        df_original (pd.DataFrame): Cleaned transaction data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Basic aggregated statistics per store and date
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
        statistics['state_bottle_cost_total'] 
        / statistics['sale_bottles'].replace(0, np.nan)
    ).replace(np.nan, 0)
    statistics['bottle_volume_ml_avg'] = (
        statistics['bottle_volume_ml_total'] 
        / statistics['sale_bottles'].replace(0, np.nan)
    ).replace(np.nan, 0)
    statistics['pack_avg'] = (
        statistics['pack_volume'] 
        / statistics['pack_number'].replace(0, np.nan)
    ).replace(np.nan, 0)

    statistics = statistics.drop(columns=[
        'state_bottle_cost_total', 
        'pack_volume', 
        'pack_number',
        'bottle_volume_ml_total'
    ])
            
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
                \rColumns: \n{statistics.columns}")
    return statistics

def get_extended_statistics(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Creates detailed store-date level statistics with distribution metrics.
    These statistics include:
    - Mean, median, min, max values for sales metrics
    - Expanded statistics for costs, volumes, and pack sizes
    - Weighted averages based on transaction volumes
    
    Args:
        df_original (pd.DataFrame): Cleaned transaction data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Detailed aggregated statistics per store and date
    """
    def calculate_weighted_median(df, value_col, weight_col):
        df_sorted = df.sort_values(['store', 'date', value_col])
        df_sorted['cum_weight'] = df_sorted.groupby(['store', 'date'])[weight_col].cumsum()
        df_sorted['total_weight'] = df_sorted.groupby(['store', 'date'])[weight_col].transform('sum')
        median_mask = df_sorted['cum_weight'] >= df_sorted['total_weight'] / 2
        return df_sorted[median_mask].groupby(['store', 'date'])[value_col].first()

    df = df_original.copy()
    logger.info("Creating additional statistics")

    df = df.assign(
        pack_number=np.ceil(df['sale_bottles'] / df['pack']),
        cost_full=df['state_bottle_cost'] * df['sale_bottles'],
        vol_full=df['bottle_volume_ml'] * df['sale_bottles'],
        pack_full=df['pack'] * np.ceil(df['sale_bottles'] / df['pack']),
        sale_bottles_abs=df['sale_bottles'].abs(),
        pack_number_abs=np.ceil(df['sale_bottles'] / df['pack']).abs()
    )

    group = df.groupby(['store', 'date'], sort=False)
    
    stats = group.agg({
        'sale_bottles':      ['mean', 'median', 'min', 'max', 'sum'],
        'sale_dollars':      ['mean', 'median', 'min', 'max'],
        'sale_liters':       ['mean', 'median', 'min', 'max'],
        'state_bottle_cost': ['min', 'max'],
        'bottle_volume_ml':  ['min', 'max'],
        'pack':              ['min', 'max'],
        'pack_number':       ['sum'],
        'cost_full':         ['sum'],
        'vol_full':          ['sum'],
        'pack_full':         ['sum']
    })
    stats.columns = [f"{col}_{func}" if func else col 
                     for col, func 
                     in stats.columns.to_flat_index()]
    stats = stats.assign(
        state_bottle_cost_mean = (
            stats['cost_full_sum'] 
            / stats['sale_bottles_sum'].replace(0, np.nan)
        ).replace(np.nan, 0),
        bottle_volume_ml_mean = (
            stats['vol_full_sum'] 
            / stats['sale_bottles_sum'].replace(0, np.nan)
        ).replace(np.nan, 0),
        pack_mean = (
            stats['pack_full_sum'] 
            / stats['pack_number_sum'].replace(0, np.nan)
        ).replace(np.nan, 0),
        state_bottle_cost_median = calculate_weighted_median(
            df, 
            'state_bottle_cost', 
            'sale_bottles_abs'
        ),
        bottle_volume_ml_median = calculate_weighted_median(
            df, 
            'bottle_volume_ml', 
            'sale_bottles_abs'
        ),
        pack_median = calculate_weighted_median(
            df, 
            'pack', 
            'pack_number_abs'
        )
    )
    stats = (stats.drop(columns=[
        'sale_bottles_sum', 
        'pack_number_sum', 
        'cost_full_sum', 
        'vol_full_sum', 
        'pack_full_sum'
    ])
    .reset_index()
    .astype({'store': df['store'].dtype})
    )
    logger.info(f"Extended statistics created.\n\
                \rShape: {stats.shape},\n\
                \rColumns: \n{stats.columns}")
    return stats

def get_item_details(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Extracts unique item details for each store-date combination.
    This includes information about:
    - Item categories
    - Product details
    - Pricing information
    
    Args:
        df_original (pd.DataFrame): Cleaned transaction data
        logger (logging.Logger): Logger instance
        
    Returns:
        pd.DataFrame: Item details aggregated per store and date
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
                \rColumns: \n{item_details.columns}")
    return item_details
