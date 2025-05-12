"""
Module for constructing LSTM datasets with train/validation/test splits.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class LSTMSequenceDataset(Dataset):
    """
    Dataset class for LSTM sequences with train/val/test splits.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        prediction_length: int = 30,
        target_col: str = 'purchase_amount',
        feature_cols: List[str] = None,
        min_history: int = 10,
        split_type: str = 'train'  # 'train', 'val', or 'test'
    ):
        """
        Initialize dataset.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            prediction_length: Length of prediction horizon
            target_col: Target column name
            feature_cols: List of feature columns to use
            min_history: Minimum number of historical points required
            split_type: Type of split ('train', 'val', or 'test')
        """
        self.df = (
            df
            .sort_values(['store','date'])
            .reset_index(drop=True)
        )
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.feature_cols = feature_cols or [col for col in df.columns if col not in ['store', 'date', target_col]]
        self.min_history = min_history
        self.split_type = split_type
        
        # build a small index map
        self.index_map: List[Tuple[str,int]] = []
        for store in self.df['store'].unique():
            store_data = (
                self.df[self.df['store'] == store]
                .reset_index(drop=True)
            )
            # skip stores with insufficient history
            if len(store_data) < self.min_history + self.sequence_length:
                continue
            max_start = len(store_data) - self.sequence_length
            for i in range(max_start):
                self.index_map.append((store, i))
        
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict:
        store, start = self.index_map[idx]
        store_data = (
            self.df[self.df['store'] == store]
            .reset_index(drop=True)
        )

        # input window
        input_end = start + self.sequence_length
        last_input_date = store_data.iloc[input_end - 1]['date'].normalize()

        features = store_data.iloc[start:input_end][self.feature_cols].values

        # make 30-day target vector
        horizon_idx = pd.date_range(
            last_input_date + pd.Timedelta(days=1),
            periods=self.prediction_length,
            freq='D'
        )
        future_rows = store_data[
            (store_data['date'] > last_input_date) &
            (store_data['date'] <= horizon_idx[-1] + pd.Timedelta(days=1))
        ]
        daily_targets = future_rows.set_index(
            future_rows['date'].dt.normalize()
        )[self.target_col]
        target_vec = daily_targets.reindex(horizon_idx, fill_value=0.0).values

        return {
            'features': torch.FloatTensor(features),
            'target': torch.FloatTensor(target_vec.astype(np.float32)),
            'store': store,
            'date': last_input_date
        }


def split_data_by_store(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_store_history: int,
    logger: logging.Logger
) -> tuple:
    """
    Split data into train, validation and test sets by time for each store.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        min_store_history: Minimum number of historical points required for a store
        logger: Logger instance
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Sort data by store and date
    df = df.sort_values(['store', 'date'])

    for col in df.columns:
        if df[col].isna().any():
            logger.warning(f"Column {col} has nan values.")
            df[col] = df[col].fillna(-1)
    
    # Filter out stores with insufficient history
    store_counts = df.groupby('store').size()
    valid_stores = store_counts[store_counts >= min_store_history].index
    df = df[df['store'].isin(valid_stores)]
    
    logger.info(f"Filtered out {len(store_counts) - len(valid_stores)} stores with insufficient history")
    logger.info(f"Remaining stores: {len(valid_stores)}")
    
    # Initialize empty DataFrames for each split
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split data for each store
    for store in df['store'].unique():
        store_data = df[df['store'] == store].copy()
        n_samples = len(store_data)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        train_dfs.append(store_data.iloc[:train_end])
        val_dfs.append(store_data.iloc[train_end:val_end])
        test_dfs.append(store_data.iloc[val_end:])
    
    # Combine splits
    train_df = pd.concat(train_dfs, axis=0)
    val_df = pd.concat(val_dfs, axis=0)
    test_df = pd.concat(test_dfs, axis=0)
    
    # Log split information
    logger.info(f"Data split by time for each store:")
    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Validation: {len(val_df)} rows")
    logger.info(f"Test: {len(test_df)} rows")
    
    return train_df, val_df, test_df

def custom_collate(batch):
    """
    Custom collate function to handle batches with timestamp fields.
    
    Args:
        batch: List of batch items (dictionaries)
        
    Returns:
        A dictionary with collated batch data
    """
    features = []
    targets = []
    timestamps = []
    
    for item in batch:
        # Convert 'date' field (Timestamp) to numeric format
        if isinstance(item.get('date'), pd.Timestamp):
            timestamps.append(item['date'].timestamp())
        else:
            timestamps.append(item.get('date', 0))
        
        # Collect feature tensors
        if isinstance(item['features'], torch.Tensor):
            features.append(item['features'].detach().clone())
        else:
            features.append(torch.tensor(item['features'], dtype=torch.float32))
        
        # Collect target tensors
        if isinstance(item['target'], torch.Tensor):
            targets.append(item['target'].detach().clone())
        else:
            targets.append(torch.tensor(item['target'], dtype=torch.float32))
    
    # Stack lists into tensors
    features = torch.stack(features)
    targets = torch.stack(targets)
    timestamps = torch.tensor(timestamps, dtype=torch.float32)
    
    # Remove redundant dimension if present (squeeze batch dim 1)
    if len(features.shape) == 3 and features.shape[1] == 1:
        features = features.squeeze(1)
    
    # If targets are (batch_size, 1), repeat to match forecast horizon length
    if len(targets.shape) == 2 and targets.shape[1] == 1:
        targets = targets.repeat(1, 5)
    
    return {
        'features': features,    # (batch_size, input_dim)
        'target': targets,       # (batch_size, prediction_length)
        'timestamp': timestamps  # (batch_size,)
    }

def create_lstm_datasets(
    df: pd.DataFrame,
    sequence_length: int = 30,
    prediction_length: int = 30,
    target_col: str = 'purchase_amount',
    feature_cols: List[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    min_store_history: int = 90,
    logger: logging.Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test datasets for LSTM model.
    
    Args:
        df: Input DataFrame
        sequence_length: Length of input sequences
        prediction_length: Length of prediction horizon
        target_col: Target column name
        feature_cols: List of feature columns to use
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        min_store_history: Minimum number of historical points required for a store
        logger: Logger instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data by store
    train_df, val_df, test_df = split_data_by_store(
        df, train_ratio, val_ratio, test_ratio, min_store_history, logger
    )
    
    # Create datasets for each split
    train_dataset = LSTMSequenceDataset(
        df=train_df,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        target_col=target_col,
        feature_cols=feature_cols,
        split_type='train'
    )
    
    val_dataset = LSTMSequenceDataset(
        df=val_df,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        target_col=target_col,
        feature_cols=feature_cols,
        split_type='val'
    )
    
    test_dataset = LSTMSequenceDataset(
        df=test_df,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        target_col=target_col,
        feature_cols=feature_cols,
        split_type='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader, test_loader 