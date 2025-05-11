"""
Script for cleaning data.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.data_utils import setup_logger, load_data
from src.config.configs import settings
from src.features.transaction_aggregation_features import (
    validate_and_clean_data
)


if __name__ == "__main__":
    """
    The main function for cleaning and validating data
    """
    parser = argparse.ArgumentParser(description='The pipeline for data cleaning before feature engineering')
    parser.add_argument('--input-file', type=str, default='data/raw/combined_data.parquet',
                        help='The path to the source data file')
    parser.add_argument('--output-file', type=str, default='data/prepared/cleaned_data.parquet',
                        help='The path to save the cleaned data')
    parser.add_argument('--log-level', 
                        type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO',
                        help='The logging level')
    parser.add_argument('--min-stores-count', type=int, default=5,
                        help='The minimum number of transactions for a store to be included')
    parser.add_argument('--max-days-between-purchases', type=int, default=200,
                        help='The maximum number of days between purchases for a store to be included')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading data: {e}")
    
    df_cleaned = validate_and_clean_data(
        df, 
        logger, 
        min_stores_count=args.min_stores_count, 
        max_days_between_purchases=args.max_days_between_purchases
    )
    
    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
       
        cleaned_data_path = Path(settings.PROJECT_ROOT, args.output_file)
        df_cleaned.to_parquet(cleaned_data_path)
        logger.info(f"Cleaned data saved in {cleaned_data_path}")
    except Exception as e:
        raise Exception(f"Error while saving cleaned data: {e}")
