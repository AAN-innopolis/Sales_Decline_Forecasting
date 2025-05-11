"""
Script for encoding features for LSTM model for predicting sales forecast.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.utils.data_utils import setup_logger, load_data, save_data
from src.utils.store_encoder import StoreEncoder
from src.utils.feature_scaler import FeatureScaler
from src.config.configs import settings

def encode_lstm_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger,
        embedding_dim: int
    ) -> pd.DataFrame:
    """
    Encode store identifiers into embeddings and add them to the DataFrame.
    
    Args:
        df_original (pd.DataFrame): Input DataFrame with 'store' column
        logger (logging.Logger): Logger object
        embedding_dim (int): Dimension of the embeddings
        
    Returns:
        pd.DataFrame: DataFrame with added store embedding columns
    """
    df = df_original.copy()
    logger.info(f"Encoding store identifiers into {embedding_dim}-dimensional embeddings")
    models_dir = Path(settings.PROJECT_ROOT, 'models/embeddings')
    store_encoder = StoreEncoder(
        embedding_dim=embedding_dim,
        logger=logger
    )
    embedding_cols = store_encoder.fit_transform(df)
    store_encoder.save(models_dir / 'store_embeddings')
    
   
    feature_scaler = FeatureScaler(logger=logger)
    df_scaled = feature_scaler.fit_transform(df)
    feature_scaler.save(models_dir / 'lstm_scaler.pkl')
    
    df_res = pd.concat([df_scaled, embedding_cols], axis=1)
    logger.info(f"Encoding successful. \
                Resulting shape: {df_res.shape}\
                Columns: {df_res.columns}")
    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for adding store embeddings to LSTM features')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/lstm_features.parquet',
                        help='Path to the input file with LSTM features')
    parser.add_argument('--output-file', type=str, default='data/prepared/lstm_features_with_embeddings.parquet',
                        help='Path to save the output file with embedded features')
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help='Dimension of the store embeddings')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Skip scaling of numerical features')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Lstm features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading lstm features: {e}")
    
    df_lstm = encode_lstm_data(
        df, 
        embedding_dim=args.embedding_dim, 
        logger=logger
    )
    
    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        df_lstm.to_parquet(Path(settings.PROJECT_ROOT, args.output_file))
        logger.info(f"Encoded lstm features saved in {Path(settings.PROJECT_ROOT, args.output_file)}")
    except Exception as e:
        raise Exception(f"Error while saving encoded lstm features: {e}")
    
