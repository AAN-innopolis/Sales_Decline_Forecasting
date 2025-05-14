import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_chronos import AutoGluonForecaster, FeaturePreprocessorChronos
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.airflow.dag_tasks.evaluation.evaluate_tft_model import load_test_dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========TFT-RELATED-STUFF====================



# ==========CHRONOS-RELATED-STUFF=================

MODEL_PATH = "models/chronos/AutogluonModels_SazeracSales" 
PREPROCESSOR_PATH = "models/chronos/feature_preprocessor_chronos.joblib"
BASE_DATA_PATH = "data/prepared/tft_features.parquet"

@st.cache_resource
def load_forecaster():
    try:
        forecaster_instance = AutoGluonForecaster(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            base_data_path=BASE_DATA_PATH
        )
        return forecaster_instance
    except Exception as e:
        st.error(f"Error initializing forecaster: {e}. Ensure paths are correct and model/preprocessor files exist.")
        return None

forecaster = load_forecaster()
AVAILABLE_STORES = forecaster.get_available_stores() if forecaster else ["Error: Model not loaded"]

# ==========CHRONOS-RELATED-STUFF=================

def predict_lstm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained LSTM model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_tft(
        store_id: str,
        date: pd.Timestamp,
        days: int
) -> pd.Series:
    """
    Generate predictions using a trained TFT model and pre-loaded test dataset.

    Args:
        store: Store identifier to forecast
        date: Last known date (start forecasting from date + 1)
        days: Number of days to forecast
        checkpoint_path: Path to trained model checkpoint
        test_dataset: Pre-loaded TimeSeriesDataSet containing test data

    Returns:
        pd.Series with predictions indexed by future dates
    """
    # Load trained model

    tft_model = TemporalFusionTransformer.load_from_checkpoint(
        "models/tft_model/best_tft_model.ckpt", map_location=device
    )
    tft_model.to(device)
    tft_model.eval()

    # logger = setup_logger(name=__name__, level="INFO")
    test_dataset, raw_df = load_test_dataset(Path("data/prepared/tft_datasets"), None)
    print(test_dataset.data)

    print("StoreId: ", store_id)
    print(type(store_id))
    print(raw_df["store"].dtype)
    raw_df["store"] = pd.to_numeric(raw_df["store"])
    print(raw_df["store"].dtype)
    print(raw_df)
    store_data = raw_df[raw_df['store'] == store_id].copy()
    print(store_data)
    store_data["store"] = store_data["store"].astype(str)


    store_data['date'] = pd.to_datetime(store_data['date'])
    # logger.info(f"Max prediction length: {test_dataset.max_prediction_length}")
    split_time_idx = store_data['time_idx'].max() \
                     - test_dataset.max_prediction_length
    # historical_data = store_data[store_data['time_idx'] <= split_time_idx][-50:]
    historical_data = store_data[store_data["date"] <= date]
    print(historical_data)
    forecast_data = store_data[store_data['time_idx'] > split_time_idx]

    store_dataset = TimeSeriesDataSet.from_dataset(test_dataset, historical_data)
    store_dataloader = store_dataset.to_dataloader(
        train=False,
        batch_size=16,
        num_workers=3
    )

    raw_output, X, actuals_output, index, decoder_lengths = tft_model.predict(
        store_dataloader,
        mode="raw",
        return_x=True,
        return_y=True,
        return_index=True,
        return_decoder_lengths=True
    )

    # central_idx = len(tft_model.loss.quantiles) // 2
    print(X)
    prediction_values = raw_output['prediction'].flatten().cpu().numpy()

    # test_dataset = TimeSeriesDataSet.load("data/prepared/tft_datasets/test_dataset.tsd")

    # Filter test data for target store

    # Find temporal boundaries
    # last_known_point = store_data[store_data["date"] <= date].iloc[-1]
    # max_time_idx = last_known_point["time_idx"]
    #
    # # Create filter for prediction window
    # time_filter = (store_data["time_idx"] > max_time_idx) & \
    #               (store_data["time_idx"] <= max_time_idx + days)

    # if time_filter.sum() < days:
    #     raise ValueError(f"Insufficient test data for {days}-day forecast")

    # Create prediction dataset subset
    # prediction_data = store_data[time_filter]
    # prediction_dataset = TimeSeriesDataSet.from_parameters(
    #     test_dataset.get_parameters(),
    #     prediction_data,
    #     predict=True,
    #     stop_randomization=True
    # )

    # Generate predictions
    # dataloader = test_dataset.to_dataloader(batch_size=32, train=False)
    predictions = prediction_values[:days]

    # Create date index for forecast
    future_dates = pd.date_range(
        start=date + pd.Timedelta(days=1),
        periods=days
    )

    return pd.Series(predictions, index=future_dates, name="sales_amount")

def predict_llm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your LLM-based forecaster
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))

def predict_chronos(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    try:
        return forecaster.predict(store_id=store, last_known_date=date, forecast_horizon_days=days)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return pd.Series([None] * days, index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))

# ===========================
# Streamlit App Layout
# ===========================

# st.set_page_config(layout="wide")
st.title("Time Series Forecast Dashboard")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("Configuration")
    model_option = st.selectbox("Model:", ["LSTM", "TFT", "LLM", "CHRONOS"])
    stores = AVAILABLE_STORES[:10]  # TODO: replace with your store list
    stores.append(10580)
    store = st.selectbox("Store:", stores)
    date = st.date_input("Last known date:", value=pd.to_datetime("2025-01-01"))
    days = st.selectbox("Forecast horizon (days):", [30], index=0)
    run = st.button("Run Forecast")


def get_historical_data(store: str, end_date: pd.Timestamp, days: int) -> pd.Series:
    """
    Get historical sales data for the specified store and time period.

    Args:
        store: Store ID
        end_date: End date for historical data
        days: Number of days of historical data to retrieve

    Returns:
        pd.Series: Historical sales data with datetime index
    """
    try:
        # Use the data already loaded in the forecaster
        if forecaster and hasattr(forecaster, 'base_df') and forecaster.base_df is not None:
            df = forecaster.base_df.copy()
        else:
            # Fallback to loading from file
            df = pd.read_parquet(BASE_DATA_PATH)
        print(df.shape)
        # Filter for the specific store
        store_data = df[df[forecaster.item_id_col] == store].copy()
        print(store_data)

        # Get the target column name from the forecaster
        # target_col = forecaster.target_col  # This should be 'sale_dollars' based on your code
        target_col = "purchase_amount"  # This should be 'sale_dollars' based on your code

        # Filter to the requested time period
        start_date = end_date - pd.Timedelta(days=days - 1)
        historical = store_data[
            (store_data[forecaster.timestamp_col] >= start_date) &
            (store_data[forecaster.timestamp_col] <= end_date)
        ]

        print(historical)

        # Convert to series with datetime index
        if not historical.empty:
            historical_series = historical.set_index(forecaster.timestamp_col)[target_col]
            return historical_series
        else:
            st.warning(f"No historical data found for store {store} in the specified date range")
            return pd.Series([], dtype=float)

    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.Series([], dtype=float)


with col2:
    st.header("Forecast Chart")
    if run:
        if model_option == "LSTM":
            preds = predict_lstm(store, pd.to_datetime(date), days)
        elif model_option == "TFT":
            preds = predict_tft(store, pd.to_datetime(date), days) #TODO FIXME
        elif model_option == "CHRONOS":
            preds = predict_chronos(store, pd.to_datetime(date), days)
        else:
            preds = predict_llm(store, pd.to_datetime(date), days)

        # Get real historical data
        real = get_historical_data(store, pd.to_datetime(date), days)

        # If historical data is empty, show a message
        if real.empty:
            st.warning("No historical data available for the selected period")
            # Create empty series with date range for plotting
            real = pd.Series([None] * days,
                             index=pd.date_range(end=pd.to_datetime(date), periods=days))

        df_plot = pd.DataFrame({"Real": real, "Predicted": preds})
        st.line_chart(df_plot)
        st.success(f"Forecast displayed for {store} from {date}.")
    else:
        st.info("Set parameters and click 'Run Forecast' to view chart.")

with col3:
    static_img = "images/putya.png"
    animated_gif = "images/putin-drinking-putin-drink.gif"
    if run:
        st.image(animated_gif, use_container_width=True)
    else:
        st.image(static_img, use_container_width=True)