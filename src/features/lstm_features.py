"""
LSTM-specific feature engineering module.
Contains functions for creating features and preparing sequences for LSTM models.
"""

import logging
import pandas as pd
import numpy as np


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
   
    lag_periods = [1, 2, 3, 4]
    window_sizes = [2, 4, 8, 12, 30, 60, 90]
    for period in lag_periods:
        df[f'prev_{period}_purchase_amount'] = df_original.groupby('store')['purchase_amount'].shift(period)
        
    
    for window in window_sizes:
        rolling_stats = (
            df_original.groupby('store')['purchase_amount']
            .rolling(
                window=window, 
                min_periods=1,
                closed='left'
            ).agg(
                ['mean', 'std', 'max', 'min', 'median']
            )
        )
        rolling_stats.columns = [
            f'hist_{stat}_{window}_purchases_amount' 
            for stat in ['mean', 'std', 'max', 'min', 'median']
        ]
        df = df.join(rolling_stats.reset_index(drop=True))
        # Calculate momentum as difference 
        # between current purchase 
        # and rolling mean of previous 'window' purchases
        df[f'purchase_momentum_{window}'] = (
            df_original['purchase_amount'] 
            - df[f'hist_mean_{window}_purchases_amount']
        )
        # Calculate percentage momentum: 
        # shows relative deviation from historical average
        # Formula: ((current/mean) - 1) * 100
        # -1 centers around 0: if current = mean, result is 0%
        df[f'purchase_momentum_pct_{window}'] = (
            (df_original['purchase_amount'] 
             / df[f'hist_mean_{window}_purchases_amount']
              .replace(0, np.nan) - 1) * 100
        )
        # Calculate average days between purchases using historical data
        df[f'hist_avg_days_between_purchases_{window}'] = (
            df_original.groupby('store')['days_since_prev_purchase']
            .rolling(window=window, min_periods=1, closed='left')
            .mean()
            .reset_index(drop=True)
        )
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