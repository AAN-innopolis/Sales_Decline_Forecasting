import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# ==========CHRONOS-RELATED-STUFF=================

MODEL_PATH = "models/chronos/AutogluonModels_SazeracSales" 
PREPROCESSOR_PATH = "models/chronos/feature_preprocessor_chronos.joblib"
BASE_DATA_PATH = "data/prepared/sazerac_sales_prepared.parquet"
LLM_BASE_DATA_PATH = "data/prepared/llm_features.parquet"

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

def predict_lstm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained LSTM model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_tft(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained TFT model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_llm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    try:
        llm_forecaster = LLMForecaster()
        
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