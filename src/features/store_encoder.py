"""
Module for encoding store identifiers into embeddings.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import logging
import pickle
from pathlib import Path


class StoreEmbedding(nn.Module):
    """
    PyTorch model for store embeddings creation.
    """
    def __init__(self, num_stores: int, embedding_dim: int = 16):
        """
        Initialize the embedding layer.
        
        Args:
            num_stores: Number of unique stores
            embedding_dim: Dimension of the embedding vectors
        """
        super(StoreEmbedding, self).__init__()
        self.store_embeddings = nn.Embedding(num_stores, embedding_dim)
        self.num_stores = num_stores
        self.embedding_dim = embedding_dim
        
    def forward(self, store_ids):
        """
        Get embeddings for store ids.
        
        Args:
            store_ids: Tensor of store ids
            
        Returns:
            Embeddings for the store ids
        """
        return self.store_embeddings(store_ids)


class StoreEncoder:
    """
    Class for encoding store identifiers into embeddings.
    """
    def __init__(self, 
                embedding_dim: int = 16,
                prefix: str = 'store_emb_',
                logger: Optional[logging.Logger] = None):
        """
        Initialize the store encoder.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            prefix: Prefix for embedding column names
            logger: Logger instance
        """
        self.embedding_dim = embedding_dim
        self.prefix = prefix
        self.logger = logger
        self.store_mapping = None
        self.model = None
        self.unique_stores = None
        self.fitted = False
        
    def _create_store_mapping(self, stores: List) -> Dict[int, int]:
        """
        Create mapping from store ids to embedding indices.
        
        Args:
            stores: List of unique store ids
            
        Returns:
            Dictionary mapping store ids to indices
        """
        return {store_id: idx for idx, store_id in enumerate(stores)}
        
    def fit(self, df: pd.DataFrame) -> 'StoreEncoder':
        """
        Fit the encoder on input DataFrame.
        
        Args:
            df: Input DataFrame containing 'store' column
            
        Returns:
            Self for method chaining
        """
        if 'store' not in df.columns:
            raise ValueError("DataFrame must contain 'store' column")
            
        if self.logger:
            self.logger.info(f"Fitting store encoder with dimension {self.embedding_dim}")
            
        # Get unique store IDs and create mapping
        self.unique_stores = sorted(df['store'].unique())
        self.store_mapping = self._create_store_mapping(self.unique_stores)
        
        # Initialize the embedding model
        num_stores = len(self.unique_stores)
        self.model = StoreEmbedding(num_stores, self.embedding_dim)
        
        self.fitted = True
        
        if self.logger:
            self.logger.info(f"Found {num_stores} unique stores")
            
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform store IDs to embeddings and add as columns.
        
        Args:
            df: Input DataFrame containing 'store' column
            
        Returns:
            DataFrame with added embedding columns
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
            
        if 'store' not in df.columns:
            raise ValueError("DataFrame must contain 'store' column")
            
        if self.logger:
            self.logger.info("Transforming store IDs to embeddings")
            
        # Convert store IDs to indices
        store_indices = df['store'].map(self.store_mapping).values
        store_indices_tensor = torch.LongTensor(store_indices)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(store_indices_tensor).numpy()
            
        # Create DataFrame with embedding columns
        embedding_cols = pd.DataFrame({
            f'{self.prefix}{i}': embeddings[:, i] for i in range(self.embedding_dim)
        }, index=df.index)
        
        if self.logger:
            self.logger.info(f"Added {self.embedding_dim} embedding columns")
            
        return embedding_cols
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder and transform store IDs to embeddings.
        
        Args:
            df: Input DataFrame containing 'store' column
            
        Returns:
            DataFrame with added embedding columns
        """
        self.fit(df)
        return self.transform(df)
        
    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save the encoder model and metadata.
        
        Args:
            output_path: Path to save the encoder
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted encoder")
            
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save the model state dict
        torch.save(self.model.state_dict(), f"{output_path}.pt")
        
        # Save metadata
        with open(f"{output_path}.pkl", 'wb') as f:
            pickle.dump({
                'store_mapping': self.store_mapping,
                'embedding_dim': self.embedding_dim,
                'prefix': self.prefix,
                'num_stores': len(self.unique_stores),
                'unique_stores': self.unique_stores
            }, f)
            
        if self.logger:
            self.logger.info(f"Store encoder saved to {output_path}")
            
    @classmethod
    def load(cls, 
            input_path: Union[str, Path], 
            logger: Optional[logging.Logger] = None) -> 'StoreEncoder':
        """
        Load encoder from disk.
        
        Args:
            input_path: Path to load the encoder from
            logger: Logger instance
            
        Returns:
            Loaded StoreEncoder instance
        """
        if logger:
            logger.info(f"Loading store encoder from {input_path}")
            
        try:
            # Load metadata
            with open(f"{input_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                
            # Create a new instance
            instance = cls(
                embedding_dim=data['embedding_dim'],
                prefix=data.get('prefix', 'store_emb_'),
                logger=logger
            )
            
            # Set attributes
            instance.store_mapping = data['store_mapping']
            instance.unique_stores = data.get('unique_stores', list(instance.store_mapping.keys()))
            
            # Initialize and load the model
            num_stores = data.get('num_stores', len(instance.unique_stores))
            instance.model = StoreEmbedding(num_stores, data['embedding_dim'])
            instance.model.load_state_dict(torch.load(f"{input_path}.pt"))
            
            instance.fitted = True
            
            if logger:
                logger.info("Store encoder loaded successfully")
                
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading store encoder: {e}")
            raise 