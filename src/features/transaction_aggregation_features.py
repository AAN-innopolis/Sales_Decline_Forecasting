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

    # if (statistics['sale_bottles'] == 0).any():
    #     logger.info("Zeros present in sale_bottles.")
    # print(statistics.columns)

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
    logger.info(f"Item details created.\
                Shape: {item_details.shape},\
                Columns: {item_details.columns}")
    return item_details
