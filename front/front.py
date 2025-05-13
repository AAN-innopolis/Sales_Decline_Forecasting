import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# Placeholder prediction functions
# Replace these with your actual model inference logic
# ===========================

def predict_lstm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained LSTM model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_tft(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your trained TFT model
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))


def predict_llm(store: str, date: pd.Timestamp, days: int) -> pd.Series:
    # TODO: integrate your LLM-based forecaster
    return pd.Series([None] * days,
                     index=pd.date_range(start=date + pd.Timedelta(days=1), periods=days))

# ===========================
# Streamlit App Layout
# ===========================

st.set_page_config(layout="wide")
st.title("Time Series Forecast Dashboard")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("Configuration")
    model_option = st.selectbox("Model:", ["LSTM", "TFT", "LLM"])
    stores = ["Store A", "Store B", "Store C"]  # TODO: replace with your store list
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
        else:
            preds = predict_llm(store, pd.to_datetime(date), days)

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