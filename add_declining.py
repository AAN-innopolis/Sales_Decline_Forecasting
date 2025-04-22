import argparse
import logging
import pandas as pd
import numpy as np
import os

from prepare_dataset import (
    load_data,
    aggregate_by_store_date,
    create_basic_features,
    create_temporal_features,
    setup_logger,
)
logger = setup_logger()

def add_is_declining_feature(input_path, output_path, target_column = "sale_dollars", decline_metric_col = "sale_dollars_significant_decrease"):
    logger.info(f"Creating 'is_declining' feature in {input_path}")

    df_original = load_data(input_path)
    df_original["date"] = pd.to_datetime(df_original["date"])
    
    df_aggregated = aggregate_by_store_date(df_original.copy()) 
    
    df_basic_feat = create_basic_features(df_aggregated.copy()) 
    
    df_tmp_feat = create_temporal_features(df_basic_feat.copy(), target = target_column)
    
    if decline_metric_col not in df_tmp_feat.columns:
        logger.error(f"Error: Column '{decline_metric_col}' was not found")
        return

    df_tmp_feat["date"] = pd.to_datetime(df_tmp_feat["date"]) 
    
    decline_mapping = df_tmp_feat[["store", "date", decline_metric_col]].copy()
    decline_mapping.rename(columns = {decline_metric_col: "is_declining"}, inplace = True)
    decline_mapping["is_declining"] = decline_mapping["is_declining"].fillna(0).astype(bool) 
    
    logger.info("Merging 'is_declining' to orignal data.")
    
    df_output = pd.merge(
        df_original,
        decline_mapping,
        on = ["store", "date"],
        how = "left"
    )
    df_output["is_declining"] = df_output["is_declining"].astype(bool)

    logger.info(f"Saving the updated dataset in {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    df_output.to_csv(output_path, index = False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, help = "Path to the original data")
    parser.add_argument("--output", type = str, help = "Path to the update data")
    parser.add_argument("--target", type = str, default = "sale_dollars", help = "Column to calculate the feature")
    parser.add_argument("--metric", type = str, default = "sale_dollars_significant_decrease",
                        help = "The feature column from prepare_dataset.py to use for the decline flag: {metric}_decrease_30d_avg, {metric}_significant_decrease etc.")
    parser.add_argument("--log-level", type = str, default = "INFO",
                        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help = "Logging level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(log_level_numeric) 
         
    add_is_declining_feature(
        input_path = args.input,
        output_path = args.output,
        target_column = args.target,
        decline_metric_col = args.metric
    )