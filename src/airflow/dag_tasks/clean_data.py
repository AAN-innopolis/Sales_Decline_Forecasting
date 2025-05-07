"""
Script for cleaning data.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import setup_logger, load_data
from src.config.configs import settings
from src.features.core_features import (
    validate_and_clean_data
)


if __name__ == "__main__":
    """
    The main function for cleaning and validating data
    """
    parser = argparse.ArgumentParser(description='The pipeline for data cleaning before feature engineering')
    parser.add_argument('--input-file', type=str, default='data/raw/sazerac_df.csv',
                        help='The path to the source data file')
    parser.add_argument('--output-file', type=str, default='data/prepared/cleaned_data.parquet',
                        help='The path to save the cleaned data')
    parser.add_argument('--log-level', 
                        type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO',
                        help='The logging level')
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_csv(Path(settings.PROJECT_ROOT, args.input_file))
        # Sort by count of records per store-date combination
        store_date_counts = df.groupby(['store', 'date']).size().reset_index(name='count')
        df = df.merge(store_date_counts, on=['store', 'date']).sort_values('count', ascending=False).drop(columns=['count']).iloc[:100000]
        logger.info(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading data: {e}")
    
    df_cleaned = validate_and_clean_data(df, logger)
    
    try:
        output_dir = Path(settings.PROJECT_ROOT, args.output_file).parent
        output_dir.mkdir(exist_ok=True, parents=True)
       
        cleaned_data_path = Path(settings.PROJECT_ROOT, args.output_file)
        df_cleaned.to_parquet(cleaned_data_path)
        logger.info(f"Cleaned data saved in {cleaned_data_path}")
    except Exception as e:
        raise Exception(f"Error while saving cleaned data: {e}")
