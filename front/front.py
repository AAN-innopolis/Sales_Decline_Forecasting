import os
from dotenv import load_dotenv
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fix PyTorch classes path issue that causes Streamlit errors
try:
    import torch.classes
    if not hasattr(torch.classes, '__path__'):
        # Create an empty list as __path__ attribute
        torch.classes.__path__ = []
    elif not isinstance(torch.classes.__path__, list):
        # If it's not a list, recreate it
        try:
            delattr(torch.classes, '__path__')
            torch.classes.__path__ = []
        except (AttributeError, TypeError):
            # If we can't delete it or there's a type error, use a more direct approach
            torch.classes.__dict__['__path__'] = []
except Exception as e:
    print(f"Warning: Could not fix torch.classes.__path__: {e}")

from src.models.train_chronos import AutoGluonForecaster, FeaturePreprocessorChronos
from src.airflow.dag_tasks.data_preparation.llm.run_query import LLMForecaster
from src.utils.data_utils import setup_logger
from src.airflow.dag_tasks.data_preparation.lstm.train_lstm_model import LitHybrid
from src.utils.feature_scaler import FeatureScaler
from src.config.configs import settings

# ==========CHRONOS-RELATED-STUFF=================

MODEL_PATH = "models/chronos/AutogluonModels_SazeracSales" 
PREPROCESSOR_PATH = "models/chronos/feature_preprocessor_chronos.joblib"
BASE_DATA_PATH = "data/prepared/sazerac_sales_prepared.parquet"
LLM_BASE_DATA_PATH = "data/prepared/llm_features.parquet"

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
api_key, base_url = os.getenv('API_KEY'), os.getenv('BASE_URL')

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

@st.cache_resource
def load_llm_df():
    try:
        df = pd.read_parquet(LLM_BASE_DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error reading llm dataset: {e}. Ensure paths are correct and files exist.")
        return None

forecaster = load_forecaster()
llm_df = load_llm_df()
AVAILABLE_STORES = forecaster.get_available_stores() if forecaster else ["Error: Model not loaded"]

# ==========CHRONOS-RELATED-STUFF=================

# ==========LSTM-RELATED-STUFF=================

logger = setup_logger(name=__name__, level="INFO")

@st.cache_data
def load_lstm_base_data(stores_length: int = 10, min_history_length: int = 300):
    df = pd.read_parquet("data/prepared/cleaned_data.parquet")
    name_to_store_dict = dict(zip(df['name'], df['store']))
    store_to_name_dict = dict(zip(df['store'], df['name']))
    store_counts = df.groupby('store').size()
    unique_ids = AVAILABLE_STORES
    return df, name_to_store_dict, store_to_name_dict, unique_ids

df, name_to_store_dict, store_to_name_dict, unique_ids = load_lstm_base_data()

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

    date = pd.to_datetime(date)
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

    print(features_seq.shape)

    # pad if few datapoints
    if features_seq.shape[0] < sequence_length:
        pad_rows = sequence_length - features_seq.shape[0]
        pad = np.zeros((pad_rows, features_seq.shape[1]))
        features_seq = np.vstack([pad, features_seq])

    # convert to tensor with batch dim
    input_tensor = torch.FloatTensor(features_seq).unsqueeze(0)  # (1, seq_len, input_dim)

    return input_tensor, actual_date

@st.cache_data
def load_model(checkpoint_path, input_dim, seq_len, horizon):
    logger.info("Entered load model")
    print(input_dim)
    model = LitHybrid.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        seq_len=seq_len,
        horizon=horizon,
        lr=1e-3  # irrelevant for inference
    )
    logger.info("Created model")
    model.eval()
    logger.info("Set model to eval")
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

@st.cache_data
def load_lstm_feature_data():
    ds_path = Path(settings.PROJECT_ROOT, "data/prepared/lstm_features_with_embeddings.parquet")
    df = pd.read_parquet(ds_path)
    logger.info("Dataset loaded.")

    # ensure o(1) access during inference
    store_data_dict = {
        store: df_store.sort_values("date").reset_index(drop=True)
        for store, df_store in df.groupby("store")
    }
    logger.info("Dataset sorted and saved.")

    return store_data_dict

store_data_dict = load_lstm_feature_data()

# ==========LSTM-RELATED-STUFF=================

def predict_lstm(input: torch.FloatTensor, store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # load model
    _, input_dim = input.shape[1:]
    model_path = Path(settings.PROJECT_ROOT, "models/attention_lstm/best.ckpt")
    model = load_model(model_path, input_dim, sequence_length, 30)
    logger.info("Model loaded")
    device = next(model.parameters()).device
    input = input.to(device)
    preds = predict(model, input)
    logger.info(f"Predicted values received.")
    return preds


def predict_tft(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained TFT model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_llm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    try:
        llm_forecaster = LLMForecaster(api_key, base_url)
        
        prompts = llm_forecaster.generate_prompts(
            df=llm_df,
            store_id=int(store),
            last_date=date.strftime("%Y-%m-%d"),
            prediction_horizon=days
        )
        if not prompts:
            raise ValueError(f"No prompts generated for store {store}.")
        
        print("PROOOMPTS", prompts)
        
        prompt_content = None
        if prompts and isinstance(prompts[0], dict):
            if 'content' in prompts[0]:
                prompt_content = prompts[0].get('content')
            elif 'prompt' in prompts[0]:
                prompt_content = prompts[0].get('prompt')
        
        print('PROMPT CONTENT: ', prompt_content)
        
        if not prompt_content:
            raise ValueError(f"Empty prompt content or invalid format: {prompts[0]}")
            
        llm_response = llm_forecaster.query_llm(prompt_content)
        print('LLM RESPONSE: ', llm_response)
        
        predictions = llm_forecaster.process_prediction(
            prediction=llm_response,
            start_date=date,
            days=days
        )
        return predictions
    except Exception as e:
        st.error(f"LLM prediction failed: {str(e)}")
        return pd.Series(
            [None] * days,
            index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days)
        )

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
    stores = unique_ids
    stores.sort()
    store = st.selectbox("Store:", stores)
    # store_id = name_to_store_dict[store]
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
        
        # Filter for the specific store
        store_data = df[df[forecaster.item_id_col] == store]
        
        # Get the target column name from the forecaster
        target_col = forecaster.target_col  # This should be 'sale_dollars' based on your code
        
        # Filter to the requested time period
        start_date = end_date - pd.Timedelta(days=days-1)
        historical = store_data[
            (store_data[forecaster.timestamp_col] >= start_date) & 
            (store_data[forecaster.timestamp_col] <= end_date)
        ]
        
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
        forecast_dates = None
        if model_option == "LSTM":
            # load input
            input, date = prepare_inference_sample(store_data_dict, store, date)
            preds = predict_lstm(input, store, pd.to_datetime(date), days)
            # plot the inferred things
            forecast_horizon = preds.shape[0]
            forecast_dates = pd.date_range(start=date , periods=forecast_horizon, freq='D')
            print(forecast_dates)
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

            scaler_path = Path(settings.PROJECT_ROOT, 'models/embeddings/lstm_scaler.pkl')
            feature_scaler = FeatureScaler.load(scaler_path, logger=logger)

            preds_original_scale = inverse_scale_purchase_amount(preds, feature_scaler)
            actual_values_original_scale = inverse_scale_purchase_amount(actual_values, feature_scaler)
            real = actual_values_original_scale
            preds = preds_original_scale
        elif model_option == "TFT":
            preds = predict_tft(store, pd.to_datetime(date), days)
        elif model_option == "CHRONOS":
            preds = predict_chronos(store, pd.to_datetime(date), days)
        else:
            preds = predict_llm(store, pd.to_datetime(date), days)

        # Get real historical data
        if model_option != "LSTM":
            real = get_historical_data(store, pd.to_datetime(date), days)

        # If historical data is empty, show a message
        # if real.empty:
          #  st.warning("No historical data available for the selected period")
            # Create empty series with date range for plotting
            # real = pd.Series([None] * days, index=pd.date_range(end=pd.to_datetime(date), periods=days))

        if forecast_dates is not None:
            df_plot = pd.DataFrame({"Real": real, "Predicted": preds, "Date": forecast_dates})
            st.line_chart(df_plot, x='Date')
        else:
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