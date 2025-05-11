"""
Script for constructing LLM prompts from preprocessed features.
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
from src.features.llm_features import prepare_llm_prompts
from src.config.configs import settings


def construct_llm_prompts(
    df: pd.DataFrame,
    logger: logging.Logger,
    prediction_horizon: int = 7,
    include_history: bool = True
) -> None:
    """
    Construct and save LLM prompts from preprocessed features.
    
    Args:
        df: Input DataFrame with LLM features
        logger: Logger instance
        prediction_horizon: Number of days to predict
        include_history: Whether to include historical context
    """
    logger.info("Starting construction of LLM prompts")
    
    # Generate prompts
    prompts = prepare_llm_prompts(
        df=df,
        prediction_horizon=prediction_horizon,
        include_history=include_history
    )
    
    # Save prompts
    prompts_dir = Path(settings.PROJECT_ROOT, 'data/prepared/llm_prompts')
    prompts_dir.mkdir(exist_ok=True, parents=True)
    
    # Save prompts as JSON
    prompts_file = prompts_dir / 'llm_prompts.json'
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    logger.info(f"LLM prompts saved in {prompts_file}")
    logger.info(f"Number of prompts generated: {len(prompts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for constructing LLM prompts')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='The logging level')
    parser.add_argument('--input-file', type=str, default='data/prepared/llm_features.parquet',
                        help='Path to the input file with LLM features')
    parser.add_argument('--prediction-horizon', type=int, default=7,
                        help='Number of days to predict')
    parser.add_argument('--include-history', action='store_true',
                        help='Include historical context in prompts')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        df = pd.read_parquet(Path(settings.PROJECT_ROOT, args.input_file))
        logger.info(f"LLM features loaded. Shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Error while loading LLM features: {e}")
    
    construct_llm_prompts(
        df=df,
        logger=logger,
        prediction_horizon=args.prediction_horizon,
        include_history=args.include_history
    ) 