"""
LSTM-specific feature engineering module.
Contains functions for creating features and preparing sequences for LSTM models.
"""

import logging
import pandas as pd
import numpy as np

from datetime import timedelta
from functools import reduce

def create_rolling_features(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Create rolling features for time series with grouping by stores.
    
    Args:
        df_original (pandas.DataFrame): Dataframe with data
        logger (logging.Logger): The logger object
        
    Returns:
        pandas.DataFrame: LSTM (rolling) features
    """
    logger.info("Creating rolling features") 
    df = df_original[['store','date']].copy()

    lookup = df_original[['store', 'date', 'purchase_amount']].copy()
    lookup.rename(columns={'purchase_amount': 'past_purchase'}, inplace=True)
   
    lag_periods = [1, 2, 3, 4]
    window_sizes = ['2D', '4D', '8D', '12D', '30D', '60D', '90D']
    for period in lag_periods:
        df[f'lag_{period}_date'] = df['date'] - timedelta(days=period)

        df = df.merge(
            lookup,
            left_on=['store', f'lag_{period}_date'],
            right_on=['store', 'date'],
            how='left'
        )

        df.rename(columns={'past_purchase': f'prev_{period}_purchase_amount'}, inplace=True)
        df.drop(columns=['date_y', f'lag_{period}_date'], inplace=True)
        df.rename(columns={'date_x': 'date'}, inplace=True)

        df[f'prev_{period}_purchase_amount'] = df[f'prev_{period}_purchase_amount'].fillna(0)

    temp = df_original[['store', 'date', 'purchase_amount', 'days_since_prev_purchase']].copy()
    temp = temp.sort_values(by=['store', 'date'])
    rolling_feature_dfs = []
    
    for window in window_sizes:

        stats = (
            temp
            .set_index('date')
            .groupby('store')['purchase_amount']
            .rolling(window=window, min_periods=1, closed='left')
            .agg(['mean', 'std', 'max', 'min', 'median'])
            .reset_index()
        )

        stats = stats.rename(columns={
            'mean': f'hist_mean_{window}_purchases_amount',
            'std': f'hist_std_{window}_purchases_amount',
            'max': f'hist_max_{window}_purchases_amount',
            'min': f'hist_min_{window}_purchases_amount',
            'median': f'hist_median_{window}_purchases_amount',
        })

        stats = stats.merge(temp[['store', 'date', 'purchase_amount']], on=['store', 'date'], how='left')

        # Calculate momentum as difference 
        # between current purchase 
        # and rolling mean of previous 'window' purchases
        stats[f'purchase_momentum_{window}'] = (
            stats['purchase_amount'] - stats[f'hist_mean_{window}_purchases_amount']
        )
        # Calculate percentage momentum: 
        # shows relative deviation from historical average
        # Formula: ((current/mean) - 1) * 100
        # -1 centers around 0: if current = mean, result is 0%
        stats[f'purchase_momentum_pct_{window}'] = (
            (stats['purchase_amount'] / stats[f'hist_mean_{window}_purchases_amount'].replace(0, np.nan) - 1) * 100
        )

        # Calculate average days between purchases using historical data
        avg_days = (
            temp
            .set_index('date')
            .groupby('store')['days_since_prev_purchase']
            .rolling(window=window, min_periods=1, closed='left')
            .mean()
            .reset_index()
            .rename(columns={'days_since_prev_purchase': f'hist_avg_days_between_purchases_{window}'})
        )

        stats = stats.merge(avg_days, on=['store', 'date'], how='left')
        stats.drop(columns=['purchase_amount'], inplace=True)

        rolling_feature_dfs.append(stats)

    all_rolling_features = reduce(lambda left, right: pd.merge(left, right, on=['store', 'date'], how='left'), rolling_feature_dfs)
    df = df.merge(all_rolling_features, on=['store', 'date'], how='left')
    logger.info(f"Rolling features created.\
                Shape: {df.shape}, \
                Columns: {df.columns}")
    return df

def prepare_lstm_sequences(df: pd.DataFrame, sequence_length: int = 30) -> np.ndarray:
    """
    Create sequences for LSTM model.
    
    Args:
        df (pd.DataFrame): Dataframe with features for LSTM
        sequence_length (int): Length of sequences to create
        
    Returns:
        np.ndarray: Array of sequences
    """
    # Sorting is critical for proper sequence creation
    df = df.sort_values(['store', 'date'])
    
    # Get store IDs
    store_ids = df['store'].unique()
    
    sequences = []
    for store_id in store_ids:
        store_data = df[df['store'] == store_id].copy()
        
        # Skip if not enough data for this store
        if len(store_data) < sequence_length + 1:
            continue
        
        # Drop non-numeric columns
        features = store_data.select_dtypes(include=[np.number])
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            seq = features.iloc[i:i+sequence_length].values
            target = features.iloc[i+sequence_length]['sale_dollars']
            sequences.append((seq, target))
    
    # Convert to numpy arrays
    X, y = [], []
    for seq, target in sequences:
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y) 