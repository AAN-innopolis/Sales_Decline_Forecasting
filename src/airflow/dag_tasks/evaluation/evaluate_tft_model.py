"""
Script for evaluating the trained Temporal Fusion Transformer (TFT) model on the test dataset.
"""

import argparse
import logging
from pathlib import Path
import sys
import traceback
from typing import Tuple
import torch
import lightning as L
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.data_utils import setup_logger
from src.config.configs import settings


def load_test_dataset(
        data_dir: Path, 
        logger: logging.Logger
    ) -> Tuple[TimeSeriesDataSet, pd.DataFrame]:
    """
    Load the test dataset.

    Args:
        data_dir: Directory where the test dataset is stored.
        logger: Logger instance.

    Returns:
        The test dataset and the raw dataframe.
    """
    logger.info(f"Loading test dataset from {data_dir}...")
    try:
        test_dataset = torch.load(data_dir / "test_dataset.pt", weights_only=False)
        raw_df = pd.read_parquet( data_dir / ".." / "tft_features.parquet" )
        categorical_features = ['store','name','address','city','zipcode','county','is_holiday', 'holiday_name']
        raw_df['purchase_amount'] = raw_df['purchase_amount'].clip(lower=0)
        raw_df[categorical_features] = raw_df[categorical_features].astype(str)
        logger.info("Test dataset loaded.")
    except FileNotFoundError:
        logger.error(f"Test dataset not found in {data_dir}.")
        raise
    return test_dataset, raw_df


def load_best_model(
        model_path: Path, 
        logger: logging.Logger
    ) -> TemporalFusionTransformer:
    """
    Load the best trained TFT model from the checkpoint.

    Args:
        model_path: Path to the best model checkpoint.
        logger: Logger instance.

    Returns:
        The loaded TFT model.
    """
    logger.info(f"Loading best model from {model_path}...")
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        logger.info("Best model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    return model


def evaluate_model(
        model: TemporalFusionTransformer, 
        test_dataset: TimeSeriesDataSet, 
        logger: logging.Logger
    ) -> None:
    """
    Evaluate the model on the test dataset and print metrics.

    Args:
        model: The trained TFT model.
        test_dataset: The test dataset.
        logger: Logger instance.
    """
    logger.info("Evaluating model on test dataset...")
    
    try:
        test_dataloader = test_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
        
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_dataloader):
                try:
                    predictions = model(x).prediction
                    actuals = y[0]
                    
                    all_predictions.append(predictions)
                    all_actuals.append(actuals)
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed {batch_idx} batches")
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    sys.exit(1)
        
        if all_predictions and all_actuals:
            all_predictions_tensor = torch.cat(all_predictions, dim=0)
            all_actuals_tensor = torch.cat(all_actuals, dim=0)
            
            central_prediction_idx = len(model.loss.quantiles) // 2
            central_predictions = all_predictions_tensor[..., central_prediction_idx]
            mae = torch.mean(torch.abs(central_predictions - all_actuals_tensor))
            
            logger.info(f"Mean Absolute Error (MAE): {mae.item()}")
        else:
            logger.warning("No valid predictions could be made")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)


def plot_predictions(
        model: TemporalFusionTransformer, 
        test_dataset: TimeSeriesDataSet, 
        raw_df: pd.DataFrame, 
        logger: logging.Logger, 
        output_dir: Path = Path("plots")
    ) -> None:
    """
    Plot predictions for the 5 longest histories (stores) in the test dataset.
    Shows historical data, predictions, and actual values during the prediction period.

    Args:
        model: The trained TFT model.
        test_dataset: The test dataset.
        raw_df: Raw dataframe containing the data.
        logger: Logger instance.
        output_dir: Directory to save plots to.
    """
    logger.info("Plotting predictions for the 5 longest histories...")

    histories = raw_df.groupby("store") \
                        .size() \
                        .nlargest(5) \
                        .index \
                        .tolist()
    for store_id in histories:
        logger.info(f"Generating predictions for store {store_id}")
        try:
            store_data = raw_df[raw_df['store'] == store_id].copy()
            store_dataset = TimeSeriesDataSet.from_dataset(test_dataset, store_data)
            store_dataloader = store_dataset.to_dataloader(
                train=False, 
                batch_size=16,
                num_workers=3
            )
            
            raw_output, X, actuals_output, index, decoder_lengths = model.predict(
                store_dataloader, 
                mode="raw",
                return_x=True,
                return_y=True, 
                return_index=True,
                return_decoder_lengths=True
            )
            central_idx = len(model.loss.quantiles) // 2
            prediction_values = raw_output['prediction'][..., central_idx].flatten().numpy()
            
            store_data['date'] = pd.to_datetime(store_data['date'])
            logger.info(f"Max prediction length: {test_dataset.max_prediction_length}")
            split_time_idx = store_data['time_idx'].max() \
                            - test_dataset.max_prediction_length
            historical_data = store_data[store_data['time_idx'] <= split_time_idx][-50:]
            forecast_data = store_data[store_data['time_idx'] > split_time_idx]
        
            fig = go.Figure()    
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['purchase_amount'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=prediction_values,
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            
            # fig.add_trace(go.Scatter(
            #     x=forecast_data['date'], 
            #     y=actuals_output.to_numpy().flatten(),
            #     mode='lines',
            #     name='Actual (Prediction Period)',
            #     line=dict(color='green', dash='dash')
            # ))
            fig.update_layout(
                title=f'Store {store_id} Sales History and Predictions',
                xaxis_title='Date',
                yaxis_title='Sales',
                legend_title='Data',
                hovermode="x unified"
            )
            plot_path = output_dir / f'store_{store_id}_predictions.html'
            fig.write_html(str(plot_path))
            logger.info(f"Plotly plot saved for store {store_id} at {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generating predictions for store {store_id}: {e}")
            logger.error(f"Trace: {traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Temporal Fusion Transformer Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "data/prepared/tft_datasets")),
        help="Directory where the test dataset is stored.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "models/tft_model/best_tft_model.ckpt")),
        help="Path to the best model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(settings.PROJECT_ROOT, "reports/figures/tft_evaluation")),
        help="Directory to save plots to.",
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level.upper())
    
    logger.info("Using CPU for evaluation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        test_dataset, raw_df = load_test_dataset(Path(args.data_dir), logger)
        model = load_best_model(Path(args.model_path), logger)
        model = model.cpu()
        
        evaluate_model(model, test_dataset, logger)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_predictions(model, test_dataset, raw_df, logger, output_dir)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1) 
