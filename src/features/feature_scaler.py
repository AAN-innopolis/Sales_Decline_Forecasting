"""
Module for feature scaling operations.
Provides FeatureScaler class for standardizing numeric features.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    Class for scaling numerical features in a dataset.
    Uses StandardScaler under the hood to standardize numeric features.
    """
    def __init__(self, 
                 exclude_columns: List[str] = None, 
                 ignore_patterns: List[str] = None,
                 logger: logging.Logger = None):
        """
        Initialize the feature scaler.
        
        Args:
            exclude_columns: Columns to explicitly exclude from scaling
            ignore_patterns: Column name patterns to exclude (e.g., ['store_emb_'])
            logger: Logger instance
        """
        self.scaler = StandardScaler()
        self.exclude_columns = exclude_columns or []
        self.ignore_patterns = ignore_patterns or []
        self.logger = logger
        self.fitted = False
        self.columns_to_scale = None
        
        # Добавим стандартные колонки для исключения
        self._add_default_exclusions()
        
    def _add_default_exclusions(self):
        """Add default columns that should be excluded from scaling."""
        default_exclusions = [
            'store', 'date', 'year', 'month', 'day', 
            'weekday', 'is_weekend', 'is_holiday', 'id'
        ]
        
        for col in default_exclusions:
            if col not in self.exclude_columns:
                self.exclude_columns.append(col)
    
    def _get_columns_to_scale(self, df: pd.DataFrame) -> List[str]:
        """
        Determine which columns should be scaled.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names to scale
        """
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclude specified columns
        columns_to_scale = [col for col in numeric_columns if col not in self.exclude_columns]
        
        # Exclude columns matching patterns
        for pattern in self.ignore_patterns:
            columns_to_scale = [col for col in columns_to_scale if not col.startswith(pattern)]
        
        if self.logger:
            self.logger.info(f"Selected {len(columns_to_scale)} numerical features for scaling")
            
        return columns_to_scale
    
    def fit(self, df: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit the scaler on the input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.columns_to_scale = self._get_columns_to_scale(df)
        
        if not self.columns_to_scale:
            if self.logger:
                self.logger.warning("No numerical features to scale")
            return self
        
        # Fit StandardScaler on selected columns
        self.scaler.fit(df[self.columns_to_scale])
        self.fitted = True
        
        if self.logger:
            self.logger.info("Scaler fitted successfully")
            
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame using the fitted scaler.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        if not self.columns_to_scale:
            if self.logger:
                self.logger.warning("No columns to scale, returning original DataFrame")
            return df.copy()
        
        # Create a copy of the input DataFrame
        df_scaled = df.copy()
        
        # Transform only selected columns
        scaled_values = self.scaler.transform(df[self.columns_to_scale])
        df_scaled[self.columns_to_scale] = scaled_values
        
        if self.logger:
            self.logger.info("Features scaled successfully")
            
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler and transform the input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        self.fit(df)
        return self.transform(df)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save the scaler to disk.
        
        Args:
            output_path: Path to save the scaler
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted scaler")
            
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save the scaler and metadata
        with open(output_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'columns_to_scale': self.columns_to_scale,
                'exclude_columns': self.exclude_columns,
                'ignore_patterns': self.ignore_patterns,
                'fitted': self.fitted
            }, f)
            
        if self.logger:
            self.logger.info(f"Feature scaler saved to {output_path}")
    
    @classmethod
    def load(cls, input_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> 'FeatureScaler':
        """
        Load a scaler from disk.
        
        Args:
            input_path: Path to load the scaler from
            logger: Logger instance
            
        Returns:
            Loaded FeatureScaler instance
        """
        if logger:
            logger.info(f"Loading feature scaler from {input_path}")
            
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
                
            # Create a new instance
            instance = cls(
                exclude_columns=data['exclude_columns'],
                ignore_patterns=data['ignore_patterns'],
                logger=logger
            )
            
            # Set loaded attributes
            instance.scaler = data['scaler']
            instance.columns_to_scale = data['columns_to_scale']
            instance.fitted = data['fitted']
            
            if logger:
                logger.info("Feature scaler loaded successfully")
                
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading feature scaler: {e}")
            raise 