"""
Script for preparing features for LLM model for predicting sales forecast.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.utils.data_utils import setup_logger
from features.static_features import (
    get_item_details,
    get_holiday_features,
    get_store_attributes,
    get_base_statistics,
    get_store_features
)
from src.config.configs import settings


def prepare_llm_data(
        df_original: pd.DataFrame, 
        logger: logging.Logger
    ) -> pd.DataFrame:
    """
    Preparation of data for LLM model.
    
    Args:
        df_original (pd.DataFrame): DataFrame with base features of transaction level information
        logger (logging.Logger): Logger object
        
    Returns:
        pd.DataFrame: DataFrame with LLM features
    """
    df = df_original.copy()
    logger.info("Starting preparation of features for LLM model")
    
    df_base = get_base_statistics(df, logger)
    df_store_attributes = get_store_attributes(df, logger)
    df_item_details = get_item_details(df, logger)
    df_res = (df_base
            .merge(
                df_store_attributes, 
                on=['store'], 
                how='left')
            .merge(
                df_item_details, 
                on=['store', 'date'], 
                how='left'
        )
    )
    logger.info(f"LLM features created. \
                Shape: {df_res.shape}, \
                Columns: {df_res.columns}")
    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The pipeline for preparing the features for LLM model')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/cleaned_data.parquet',
                        help='The path to the file with cleaned data')
    parser.add_argument('--output-file', type=str, default='data/prepared/llm_features.parquet',
                        help='The path to save the features for LLM model')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Base features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading base features: {e}")
    
    df_llm = prepare_llm_data(df, logger)

    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        df_llm.to_parquet(Path(settings.PROJECT_ROOT, args.output_file))
        logger.info(f"LLM features saved in {Path(settings.PROJECT_ROOT, args.output_file)}")
    except Exception as e:
        raise Exception(f"Error while saving LLM features: {e}")
    