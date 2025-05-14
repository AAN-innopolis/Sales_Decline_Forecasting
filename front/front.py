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
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========TFT-RELATED-STUFF====================


features = {
    'static': [
        'store', 'zipcode', 'lon', 'lat', 'store_size'
    ],
    'observed': [
        'purchased_bottles', 'purchased_liters', 'transaction_count', 'unique_categories', 'unique_items',
        'average_state_bottle_cost', 'average_bottle_volume', 'average_pack', 'sale_bottles_mean',
        'sale_bottles_median', 'sale_bottles_min', 'sale_bottles_max', 'sale_dollars_mean', 'sale_dollars_median',
        'sale_dollars_min', 'sale_dollars_max', 'sale_liters_mean', 'sale_liters_median', 'sale_liters_min',
        'sale_liters_max', 'state_bottle_cost_min', 'state_bottle_cost_max', 'bottle_volume_ml_min',
        'bottle_volume_ml_max', 'pack_min', 'pack_max', 'state_bottle_cost_mean', 'bottle_volume_ml_mean',
        'pack_mean', 'state_bottle_cost_median', 'bottle_volume_ml_median', 'pack_median',
        'days_since_prev_purchase', 'avg_price_per_bottle', 'avg_price_per_liter', 'avg_items_per_transaction',
        'avg_transaction_value', 'store_avg_sales', 'store_avg_transactions', 'store_avg_items', 'city_avg_sales',
        'county_avg_sales', 'store_to_city_sales_ratio', 'store_to_county_sales_ratio',
        'hist_mean_7D_purchases_amount', 'hist_std_7D_purchases_amount', 'hist_max_7D_purchases_amount',
        'hist_min_7D_purchases_amount', 'hist_median_7D_purchases_amount', 'purchase_momentum_7D',
        'purchase_momentum_pct_7D', 'hist_avg_days_between_purchases_7D', 'hist_mean_14D_purchases_amount',
        'hist_std_14D_purchases_amount', 'hist_max_14D_purchases_amount', 'hist_min_14D_purchases_amount',
        'hist_median_14D_purchases_amount', 'purchase_momentum_14D', 'purchase_momentum_pct_14D',
        'hist_avg_days_between_purchases_14D', 'hist_mean_21D_purchases_amount', 'hist_std_21D_purchases_amount',
        'hist_max_21D_purchases_amount', 'hist_min_21D_purchases_amount', 'hist_median_21D_purchases_amount',
        'purchase_momentum_21D', 'purchase_momentum_pct_21D', 'hist_avg_days_between_purchases_21D',
        'hist_mean_30D_purchases_amount', 'hist_std_30D_purchases_amount', 'hist_max_30D_purchases_amount',
        'hist_min_30D_purchases_amount', 'hist_median_30D_purchases_amount', 'purchase_momentum_30D',
        'purchase_momentum_pct_30D', 'hist_avg_days_between_purchases_30D', 'hist_mean_60D_purchases_amount',
        'hist_std_60D_purchases_amount', 'hist_max_60D_purchases_amount', 'hist_min_60D_purchases_amount',
        'hist_median_60D_purchases_amount', 'purchase_momentum_60D', 'purchase_momentum_pct_60D',
        'hist_avg_days_between_purchases_60D', 'hist_mean_90D_purchases_amount', 'hist_std_90D_purchases_amount',
        'hist_max_90D_purchases_amount', 'hist_min_90D_purchases_amount', 'hist_median_90D_purchases_amount',
        'purchase_momentum_90D', 'purchase_momentum_pct_90D', 'hist_avg_days_between_purchases_90D'
    ],
    'known_future': [
        'time_idx', 'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos',
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'week_of_year_sin', 'week_of_year_cos', 'year',
        'is_holiday', 'days_to_nearest_holiday'
    ]
}


@st.cache_resource
def load_tft_tf_resources():
    """Load TFT-TF model with cross-version compatibility"""
    try:
        model_path = Path("models/tft_tf_model")
        data_dir = Path("data/prepared/tft_tf_datasets")

        # 1. Configure legacy Keras behavior for compatibility
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        from tensorflow.keras.models import load_model

        # 2. Load model with explicit custom object scope
        with tf.keras.utils.custom_object_scope({'TFSMLayer': tf.keras.layers.TFSMLayer}):
            model = load_model(str(model_path))

        # 3. Verify input signatures directly from loaded model
        input_details = {
            'static': model.inputs[0].shape.as_list(),
            'observed': model.inputs[1].shape.as_list(),
            'known': model.inputs[2].shape.as_list()
        }

        # 4. Load dataset with version-aware casting
        with np.load(data_dir / 'test_data.npz', allow_pickle=True) as data:
            test_data = {
                'static': data['static'].astype(np.float32),
                'observed': data['X_seq'].astype(np.float32),
                'known': data['known'].astype(np.float32),
                'target': data['y']
            }

        # 5. Load raw dataframe with backward compatibility
        raw_df = pd.read_parquet("data/prepared/tft_features.parquet")

        return model, test_data, raw_df

    except Exception as e:
        st.error(f"Error loading TFT-TF resources: {str(e)}")
        return None, None, None

# checkpoint_path = "models/tft_model/best_tft_model.ckpt"  # Update with your path
# tft_model = TemporalFusionTransformer.load_from_checkpoint(
#     checkpoint_path, map_location=device
# )
# tft_model.to(device)
# tft_model.eval()

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


def predict_tft_tf(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    model, test_data, raw_df = load_tft_tf_resources()

    try:
        # Find matching data with version-agnostic indexing
        mask = (raw_df['store'] == int(store)) & (raw_df['date'] == date)
        store_data = raw_df[mask]

        if store_data.empty:
            st.error(f"No data for store {store} on {date}")
            return pd.Series([], dtype='float64')

        sample_idx = store_data.index[0]

        # Prepare inputs with dynamic shape handling
        static_input = test_data['static'][sample_idx:sample_idx + 1]
        observed_input = test_data['observed'][sample_idx:sample_idx + 1]
        known_input = test_data['known'][sample_idx:sample_idx + 1]

        # Generate prediction with version-compatible execution
        prediction = model.predict([static_input, observed_input, known_input])

        # Handle different output formats between versions
        if isinstance(prediction, dict):
            preds = prediction['output'].flatten()[:days]
        else:
            preds = prediction.flatten()[:days]

        future_dates = pd.date_range(
            start=date + pd.Timedelta(days=1),
            periods=days
        )

        return pd.Series(preds, index=future_dates, name="sales_amount")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return pd.Series([], dtype='float64')


def predict_tft(
        store: str,
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

    test_dataset = TimeSeriesDataSet.load("data/prepared/tft_datasets/test_dataset.tsd")

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
    dataloader = test_dataset.to_dataloader(batch_size=32, train=False)
    predictions = tft_model.predict(dataloader).cpu().numpy().flatten()[:days]

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
    store = st.selectbox("Store:", stores)
    date = st.date_input("Last known date:", value=pd.to_datetime("2025-01-01"))
    days = st.selectbox("Forecast horizon (days):", [30], index=0)
    run = st.button("Run Forecast")

with col2:
    st.header("Forecast Chart")
    if run:
        if model_option == "LSTM":
            preds = predict_lstm(store, pd.to_datetime(date), days)
        elif model_option == "TFT":
            preds = predict_tft(store, pd.to_datetime(date), days)
        elif model_option == "CHRONOS":
            preds = predict_chronos(store, pd.to_datetime(date), days)
        else:
            preds = predict_llm(store, pd.to_datetime(date), days)

        # TODO: complete real 
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