"""
Module for encoding features for Temporal Fusion Transformer model.
"""

import pandas as pd
import numpy as np
import torch
import pickle
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet


class TFTFeatureEncoder:
    """
    Feature encoder for Temporal Fusion Transformer models
    """
    
    def __init__(
        self,
        max_encoder_length: int,
        max_prediction_length: int,
        target_col: str,
        time_idx_col: str = "time_idx",
        group_ids: List[str] = None,
        static_categoricals: List[str] = None,
        static_reals: List[str] = None,
        time_varying_known_categoricals: List[str] = None,
        time_varying_known_reals: List[str] = None,
        time_varying_unknown_categoricals: List[str] = None,
        time_varying_unknown_reals: List[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the TFT feature encoder.
        
        Args:
            max_encoder_length: Maximum lookback window length
            max_prediction_length: Maximum prediction horizon
            target_col: Target column name
            time_idx_col: Time index column name
            group_ids: List of ID columns that identify a time series
            static_categoricals: List of static categorical features
            static_reals: List of static real-valued features
            time_varying_known_categoricals: List of time-varying known categorical features
            time_varying_known_reals: List of time-varying known real-valued features
            time_varying_unknown_categoricals: List of time-varying unknown categorical features
            time_varying_unknown_reals: List of time-varying unknown real-valued features
            logger: Logger instance
        """
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target_col = target_col
        self.time_idx_col = time_idx_col
        self.group_ids = group_ids or []
        self.static_categoricals = static_categoricals or []
        self.static_reals = static_reals or []
        self.time_varying_known_categoricals = time_varying_known_categoricals or []
        self.time_varying_known_reals = time_varying_known_reals or []
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals or []
        self.time_varying_unknown_reals = time_varying_unknown_reals or []
        self.logger = logger or logging.getLogger(__name__)
        
        # Record column mappings for feature groups
        self.feature_groups = {
            'static_categoricals': self.static_categoricals,
            'static_reals': self.static_reals,
            'time_varying_known_categoricals': self.time_varying_known_categoricals,
            'time_varying_known_reals': self.time_varying_known_reals,
            'time_varying_unknown_categoricals': self.time_varying_unknown_categoricals,
            'time_varying_unknown_reals': self.time_varying_unknown_reals,
        }
        
        # Save the list of all feature names
        self.all_features = (
            self.group_ids + 
            self.static_categoricals + 
            self.static_reals + 
            self.time_varying_known_categoricals + 
            self.time_varying_known_reals + 
            self.time_varying_unknown_categoricals + 
            self.time_varying_unknown_reals
        )
        
        # Ensure target column is included in features
        if self.target_col not in self.all_features:
            self.time_varying_unknown_reals.append(self.target_col)
            self.all_features.append(self.target_col)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and encode a dataframe for TFT.
        
        Args:
            df: Input dataframe with features
            
        Returns:
            DataFrame with encoded features
        """
        # Check that all required columns are present
        missing_cols = set(self.all_features) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure time index column exists
        if self.time_idx_col not in df.columns:
            self.logger.info(f"Creating time index column: {self.time_idx_col}")
            df = self._create_time_index(df)
        
        # Keep only needed columns
        needed_cols = [self.time_idx_col] + self.all_features
        df = df[needed_cols].copy()
        
        # Ensure group_ids are categorical
        for col in self.group_ids:
            df[col] = df[col].astype('category')
        
        # Validate that we have the target column
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col} not found in dataframe")
        
        # Convert categorical columns to the correct dtype
        all_categorical_cols = (
            self.static_categoricals + 
            self.time_varying_known_categoricals + 
            self.time_varying_unknown_categoricals
        )
        
        for col in all_categorical_cols:
            df[col] = df[col].astype('category')
        
        self.logger.info(f"DataFrame encoded successfully. Shape: {df.shape}")
        return df
    
    def _create_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a time index column for the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with time index column
        """
        if 'date' in df.columns:
            # Sort by group_id and date
            sort_cols = self.group_ids + ['date']
            df = df.sort_values(sort_cols)
            
            # Create time index per group
            df[self.time_idx_col] = df.groupby(self.group_ids).cumcount()
            self.logger.info(f"Created time index from date column")
        else:
            raise ValueError("Cannot create time index: 'date' column not found")
        
        return df
    
    def create_tft_dataset_parameters(self) -> Dict[str, Any]:
        """
        Create parameters for TimeSeriesDataSet initialization.
        
        Returns:
            Dictionary of parameters for TimeSeriesDataSet
        """
        params = {
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "time_idx": self.time_idx_col,
            "target": self.target_col,
            "group_ids": self.group_ids,
            "static_categoricals": self.static_categoricals,
            "static_reals": self.static_reals,
            "time_varying_known_categoricals": self.time_varying_known_categoricals,
            "time_varying_known_reals": self.time_varying_known_reals,
            "time_varying_unknown_categoricals": self.time_varying_unknown_categoricals,
            "time_varying_unknown_reals": self.time_varying_unknown_reals,
            # Default parameters
            "add_relative_time_idx": True,
            "add_target_scales": True,
            "add_encoder_length": True,
            "randomize_length": None,
            "target_normalizer": None  # Will be determined during training
        }
        
        return params
    
    def save(self, path: Union[str, Path]):
        """
        Save the encoder to a file.
        
        Args:
            path: Path to save the encoder
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"TFT Feature Encoder saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TFTFeatureEncoder':
        """
        Load a saved encoder from a file.
        
        Args:
            path: Path to load the encoder from
            
        Returns:
            Loaded TFTFeatureEncoder instance
        """
        with open(path, 'rb') as f:
            encoder = pickle.load(f)
        
        if not isinstance(encoder, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        
        return encoder 