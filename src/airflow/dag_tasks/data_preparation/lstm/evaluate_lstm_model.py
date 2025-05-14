import pandas as pd
import numpy as np
import argparse
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.config.configs import settings
from src.airflow.dag_tasks.data_preparation.lstm.train_lstm_model import LitHybrid
from src.models.attention_lstm import HybridLSTMAttn
from src.utils.data_utils import setup_logger
from src.utils.feature_scaler import FeatureScaler

sequence_length = 30
embedding_size = 16
feature_cols = [
        # Time features
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        
        # Rolling statistics
        f'hist_mean_{sequence_length}D_purchases_amount',
        f'hist_std_{sequence_length}D_purchases_amount',
        f'hist_max_{sequence_length}D_purchases_amount',
        f'hist_min_{sequence_length}D_purchases_amount',
        f'hist_median_{sequence_length}D_purchases_amount',
        
        # Momentum features
        f'purchase_momentum_{sequence_length}D',
        f'purchase_momentum_pct_{sequence_length}D',

        # Store embeddings
        *[f'store_emb_{emb}' for emb in range(embedding_size)]
    ]

def prepare_inference_sample(df_dict: dict, store: int = 2106, date: pd.Timestamp = pd.Timestamp('2023-01-01'),
                             sequence_length: int = sequence_length, feature_cols: list = feature_cols, prediction_length: int = 30):
    """
    Mimics __getitem__ logic to create one inference sample.
    """
    store_data = df_dict[store]

    valid_data = store_data[store_data["date"] <= date]

    if valid_data.empty:
        logger.error(f"No data available for store {store} before or on {date}")
        return None

    # last available date becomes the actual base date
    actual_date = valid_data["date"].iloc[-1]
    target_idx = valid_data.index[-1]

    # extract feature sequence
    start_row = max(0, target_idx - sequence_length)
    features_seq = store_data.iloc[start_row:target_idx][feature_cols].values

    # pad if few datapoints
    if features_seq.shape[0] < sequence_length:
        pad_rows = sequence_length - features_seq.shape[0]
        pad = np.zeros((pad_rows, features_seq.shape[1]))
        features_seq = np.vstack([pad, features_seq])

    # convert to tensor with batch dim
    input_tensor = torch.FloatTensor(features_seq).unsqueeze(0)  # (1, seq_len, input_dim)

    return input_tensor, actual_date

def load_model(checkpoint_path, input_dim, seq_len, horizon):
    model = LitHybrid.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        seq_len=seq_len,
        horizon=horizon,
        lr=1e-3  # irrelevant for inference
    )
    model.eval()
    return model

def predict(model, input_tensor):
    with torch.no_grad():
        prediction = model(input_tensor)  # shape: (1, horizon, 1) or (1, horizon)
    return prediction.squeeze(0).cpu().numpy()

def inverse_scale_purchase_amount(values: np.ndarray, feature_scaler: FeatureScaler) -> np.ndarray:
    """
    Inverse scale 'purchase_amount' values using the provided FeatureScaler.
    
    Args:
        values (np.ndarray): Array of scaled purchase_amount values (shape: (n_samples,))
        feature_scaler (FeatureScaler): The fitted FeatureScaler object
    
    Returns:
        np.ndarray: Inverse-scaled purchase_amount values in original scale
    """
    # Prepare dummy DataFrame with correct columns
    dummy_data = pd.DataFrame(0, index=range(len(values)), columns=feature_scaler.columns_to_scale)

    # Put values into 'purchase_amount' column
    dummy_data['purchase_amount'] = values

    # Inverse transform
    inverse_scaled = feature_scaler.scaler.inverse_transform(dummy_data)

    # Extract only purchase_amount column back
    inverse_purchase_amount = pd.DataFrame(inverse_scaled, columns=feature_scaler.columns_to_scale)['purchase_amount'].values

    return inverse_purchase_amount

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate HybridLSTMAttn model")
    parser.add_argument("--checkpoint", type=str, help="Path to best checkpoint", default="models/attention_lstm/best.ckpt")
    parser.add_argument("--dataset-dir", type=str, default="data/prepared/lstm_features_with_embeddings.parquet")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.log_level)
    store = 2106
    date = pd.Timestamp('2023-01-01')

    # load full dataset
    ds_path = Path(settings.PROJECT_ROOT, args.dataset_dir)
    df = pd.read_parquet(ds_path)
    logger.info("Dataset loaded.")

    # ensure o(1) access during inference
    store_data_dict = {
        store: df_store.sort_values("date").reset_index(drop=True)
        for store, df_store in df.groupby("store")
    }
    logger.info("Dataset sorted and saved.")

    # load input
    input, date = prepare_inference_sample(store_data_dict)

    # load model
    _, input_dim = input.shape[1:]
    model_path = Path(settings.PROJECT_ROOT, args.checkpoint)
    model = load_model(model_path, input_dim, sequence_length, 30)
    logger.info("Model loaded")

    # do inference
    device = next(model.parameters()).device
    input = input.to(device)
    preds = predict(model, input)
    logger.info(f"Predicted values received.")

    # plot the inferred things
    forecast_horizon = preds.shape[0]
    forecast_dates = pd.date_range(start=date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

    # get the real values
    actual_values = []

    store_data = store_data_dict[store]

    for forecast_date in forecast_dates:
        match = store_data[store_data["date"] == forecast_date]
        if not match.empty:
            actual_values.append(match["purchase_amount"].values[0])
        else:
            actual_values.append(0.0)  # No data → assume zero sales

    actual_values = np.array(actual_values)

    # upscale the preds and actual values
    scaler_path = Path(settings.PROJECT_ROOT, 'models/embeddings/lstm_scaler.pkl')
    feature_scaler = FeatureScaler.load(scaler_path, logger=logger)

    preds_original_scale = inverse_scale_purchase_amount(preds, feature_scaler)
    actual_values_original_scale = inverse_scale_purchase_amount(actual_values, feature_scaler)

    # plot the comparison
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, preds_original_scale, marker='o', linestyle='-', color='blue', label='Predicted Sales')
    plt.plot(forecast_dates, actual_values_original_scale, marker='x', linestyle='--', color='red', label='Actual Sales')
    plt.title(f"Sales Forecast for Store {store} starting from {date.date()}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Purchase Amount")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()