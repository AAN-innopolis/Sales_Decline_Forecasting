import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib # For saving/loading the preprocessor
import holidays

from typing import List, Dict
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# --- Feature Definitions (Copied from original script) ---
STATIC_CATEGORICAL = [
    "name", "address", "city", "zipcode", "county",
]

STATIC_NUM = [
    "lon", "lat",
]

DYNAMIC = [
    "day_of_week_sin", "day_of_week_cos",
    "day_of_month_sin", "day_of_month_cos",
    "month_sin", "month_cos",
    "quarter_sin", "quarter_cos",
    "week_of_year_sin", "week_of_year_cos",
    "year",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
]

KNOWN_DYNAMIC_FEATURES = [ # These are features known in advance for the future
    "day_of_week_sin", "day_of_week_cos",
    "day_of_month_sin", "day_of_month_cos",
    "month_sin", "month_cos",
    "quarter_sin", "quarter_cos",
    "week_of_year_sin", "week_of_year_cos",
    "year",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
]

# Add historical aggregate features to DYNAMIC (used for training, not necessarily all known for future)
for w in [2, 4, 8, 12, 30, 60, 90]:
    for stat in ["mean", "std", "max", "min", "median"]:
        DYNAMIC.append(f"hist_{stat}_{w}_purchases_sale_dollars")
    DYNAMIC.extend([
        f"purchase_momentum_{w}",
        f"purchase_momentum_pct_{w}",
        f"hist_avg_days_between_purchases_{w}",
    ])

TARGET = ["sale_dollars"]
ITEM_ID_COL = "store"
TIMESTAMP_COL = "date"
TARGET_COL_NAME = "sale_dollars" # Original target column name

# --- FeaturePreprocessorChronos Class (Copied and slightly adapted) ---
class FeaturePreprocessorChronos:
    def __init__(self, static_cat_cols: List[str], static_num_cols: List[str],
                 dynamic_cols: List[str], target_cols: List[str]):
        self.static_cat_cols = static_cat_cols
        self.static_num_cols = static_num_cols
        self.dynamic_cols = dynamic_cols # All dynamic cols seen during training
        self.target_cols = target_cols

        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        # Chronos handles target scaling internally if configured, so no target_scaler here.

        self.cat_encodings = {}
        self.padding_values = {}

    def fit(self, df: pd.DataFrame):
        for col in self.static_cat_cols:
            # Ensure NaN is not treated as a category for encoding, handle it separately or let map handle it
            unique_vals = df[col].dropna().unique()
            self.cat_encodings[col] = {val: idx + 2 for idx, val in enumerate(unique_vals)} # 0 for padding, 1 for unknown
            self.padding_values[col] = 0

        if self.static_num_cols:
            self.static_scaler.fit(df[self.static_num_cols].fillna(0)) # Fill NaNs before fitting scaler
        if self.dynamic_cols:
            self.dynamic_scaler.fit(df[self.dynamic_cols].fillna(0)) # Fill NaNs before fitting scaler
        # No target scaler fitting here for Chronos

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.static_cat_cols:
            df[col] = df[col].map(lambda x: self.cat_encodings[col].get(x, 1)) # Map unknown to 1
            df[col] = df[col].fillna(self.padding_values[col]) # Fill NaNs (originally padding)

        if self.static_num_cols:
            # Fill NaNs before transforming, consistent with fit
            df[self.static_num_cols] = self.static_scaler.transform(df[self.static_num_cols].fillna(0))
            # Scaler might produce NaNs if a column was all NaN during fit; ensure these are 0
            df[self.static_num_cols] = np.nan_to_num(df[self.static_num_cols], nan=0.0)


        if self.dynamic_cols:
            # Fill NaNs before transforming, consistent with fit
            df[self.dynamic_cols] = self.dynamic_scaler.transform(df[self.dynamic_cols].fillna(0))
            df[self.dynamic_cols] = np.nan_to_num(df[self.dynamic_cols], nan=0.0)

        # Target is not transformed by this preprocessor for Chronos
        if self.target_cols:
            for col in self.target_cols:
                if col in df.columns:
                    df[col] = df[col].copy() # Ensure it's not a view
                else: # If target is missing (e.g. future data), don't try to access it
                    pass
        return df

def generate_future_known_features(df: pd.DataFrame, known_dynamic_feature_names: List[str]) -> pd.DataFrame:
    """
    Generates time-based features for a future dataframe.

    Args:
        df: DataFrame with 'timestamp' and 'item_id' columns.
        known_dynamic_feature_names: List of known dynamic feature names to generate.

    Returns:
        DataFrame with added known dynamic features.
    """
    df_out = df[['item_id', 'timestamp']].copy() # Start with item_id and timestamp
    df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
    dt = df_out['timestamp'].dt

    # Basic time features
    df_out['day_of_week'] = dt.dayofweek
    df_out['day_of_month'] = dt.day
    df_out['month'] = dt.month
    df_out['quarter'] = dt.quarter
    df_out['week_of_year'] = dt.isocalendar().week.astype(int)
    df_out['year'] = dt.year

    # Cyclical features
    if 'day_of_week_sin' in known_dynamic_feature_names:
        df_out['day_of_week_sin'] = np.sin(2 * np.pi * df_out['day_of_week'] / 7)
    if 'day_of_week_cos' in known_dynamic_feature_names:
        df_out['day_of_week_cos'] = np.cos(2 * np.pi * df_out['day_of_week'] / 7)
    if 'day_of_month_sin' in known_dynamic_feature_names:
        df_out['day_of_month_sin'] = np.sin(2 * np.pi * df_out['day_of_month'] / dt.days_in_month)
    if 'day_of_month_cos' in known_dynamic_feature_names:
        df_out['day_of_month_cos'] = np.cos(2 * np.pi * df_out['day_of_month'] / dt.days_in_month)
    if 'month_sin' in known_dynamic_feature_names:
        df_out['month_sin'] = np.sin(2 * np.pi * df_out['month'] / 12)
    if 'month_cos' in known_dynamic_feature_names:
        df_out['month_cos'] = np.cos(2 * np.pi * df_out['month'] / 12)
    if 'quarter_sin' in known_dynamic_feature_names:
        df_out['quarter_sin'] = np.sin(2 * np.pi * df_out['quarter'] / 4)
    if 'quarter_cos' in known_dynamic_feature_names:
        df_out['quarter_cos'] = np.cos(2 * np.pi * df_out['quarter'] / 4)
    if 'week_of_year_sin' in known_dynamic_feature_names:
        df_out['week_of_year_sin'] = np.sin(2 * np.pi * df_out['week_of_year'] / 52.14) # Use 52 or 53 based on year if exactness is critical
    if 'week_of_year_cos' in known_dynamic_feature_names:
        df_out['week_of_year_cos'] = np.cos(2 * np.pi * df_out['week_of_year'] / 52.14)

    # Weekend
    if 'is_weekend' in known_dynamic_feature_names:
        df_out['is_weekend'] = df_out['day_of_week'].isin([5, 6]).astype(int)

    # Holiday features
    if 'is_holiday' in known_dynamic_feature_names or 'days_to_nearest_holiday' in known_dynamic_feature_names:
        min_year, max_year = df_out['year'].min(), df_out['year'].max()
        # Ensure years are valid for holidays package
        if pd.isna(min_year) or pd.isna(max_year): # Handle empty df_out case
             if 'is_holiday' in known_dynamic_feature_names: df_out['is_holiday'] = 0
             if 'days_to_nearest_holiday' in known_dynamic_feature_names: df_out['days_to_nearest_holiday'] = 999
        else:
            us_holidays = holidays.US(years=range(int(min_year), int(max_year) + 1))
            df_out['date_only'] = df_out['timestamp'].dt.date
            if 'is_holiday' in known_dynamic_feature_names:
                df_out['is_holiday'] = df_out['date_only'].apply(lambda date_obj: 1 if date_obj in us_holidays else 0)

            if 'days_to_nearest_holiday' in known_dynamic_feature_names:
                holiday_dates_ts = pd.to_datetime(pd.Series(sorted(us_holidays.keys())))
                if not holiday_dates_ts.empty:
                    def days_diff_np(date, holiday_dates_ts_arg):
                        diffs = np.abs((holiday_dates_ts_arg - pd.Timestamp(date)).dt.days)
                        return np.min(diffs) if not diffs.empty else np.inf
                    df_out['days_to_nearest_holiday'] = df_out['date_only'].apply(days_diff_np, holiday_dates_ts_arg=holiday_dates_ts)
                else:
                    df_out['days_to_nearest_holiday'] = np.inf
                df_out['days_to_nearest_holiday'] = df_out['days_to_nearest_holiday'].replace(np.inf, 999).astype(int)

    # Keep only the required known dynamic features + item_id and timestamp
    keep_cols = ['item_id', 'timestamp'] + [col for col in known_dynamic_feature_names if col in df_out.columns]
    return df_out[keep_cols]

class AutoGluonForecaster:
    def __init__(self, model_path: str, preprocessor_path: str, base_data_path: str):
        print(f"Loading AutoGluon model from: {model_path}")
        self.predictor = TimeSeriesPredictor.load(model_path)
        print(f"Loading preprocessor from: {preprocessor_path}")
        self.preprocessor: FeaturePreprocessorChronos = joblib.load(preprocessor_path)
        
        print(f"Loading base data from: {base_data_path}")
        self.base_df_orig = pd.read_parquet(base_data_path)

        # Define these attributes BEFORE calling _preprocess_base_df
        self.item_id_col = ITEM_ID_COL
        self.timestamp_col = TIMESTAMP_COL
        self.target_col = TARGET_COL_NAME # Original target name for final series
        self.autogluon_target_col = self.predictor.target # Target name AG expects

        # Now call _preprocess_base_df
        self.base_df = self._preprocess_base_df(self.base_df_orig.copy())

        self.static_cat_cols = self.preprocessor.static_cat_cols
        self.static_num_cols = self.preprocessor.static_num_cols
        self.known_dynamic_features = KNOWN_DYNAMIC_FEATURES # From global scope

        self.model_prediction_length = self.predictor.prediction_length
        self.model_freq = self.predictor.freq

    def _preprocess_base_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply same initial cleaning as in training
        store_counts = df.groupby(self.item_id_col).size() # Use self.item_id_col
        valid_stores = store_counts[store_counts >= 70].index
        df = df[df[self.item_id_col].isin(valid_stores)] # Use self.item_id_col
        
        if self.target_col in df.columns: # Use self.target_col
            df[self.target_col] = np.where(df[self.target_col] < 0, 0, df[self.target_col])
            if not df.empty:
                # Use self.base_df_orig for quantile calculation, filtered by valid_stores
                # to be consistent with how df is being filtered.
                # Also ensure item_id_col and target_col are used correctly here.
                quantile_999 = self.base_df_orig[
                    self.base_df_orig[self.item_id_col].isin(valid_stores) # Use self.item_id_col
                ][self.target_col].quantile(0.999) # Use self.target_col

                df[self.target_col] = np.where(df[self.target_col] > quantile_999, # Use self.target_col
                                             quantile_999,
                                             df[self.target_col]) # Use self.target_col
        
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col]) # Use self.timestamp_col
        return df

    def get_available_stores(self) -> List[str]:
        return sorted(self.base_df[self.item_id_col].unique().tolist()) # Use self.item_id_col

    # --- AutoGluonForecaster Class (predict method only) ---
    def predict(self, store_id: str, last_known_date: pd.Timestamp, forecast_horizon_days: int) -> pd.Series:
        last_known_date = pd.to_datetime(last_known_date)

        # 1. Prepare historical data for the given store
        history_df_store_orig = self.base_df[
            (self.base_df[self.item_id_col] == store_id) &
            (self.base_df[self.timestamp_col] <= last_known_date)
        ].copy()

        if history_df_store_orig.empty:
            print(f"No historical data found for store {store_id} up to {last_known_date}.")
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
            return pd.Series([np.nan] * forecast_horizon_days, index=future_dates, name=self.target_col)

        history_df_transformed = self.preprocessor.transform(history_df_store_orig)
        
        history_df_ag = history_df_transformed.rename(columns={
            self.item_id_col: "item_id",       
            self.timestamp_col: "timestamp",   
            self.target_col: self.autogluon_target_col 
        })
        
        static_cols_for_ag = ["item_id"] + self.static_cat_cols + self.static_num_cols
        current_static_cols = [col for col in static_cols_for_ag if col in history_df_ag.columns]
        # Ensure 'item_id' is present before groupby if current_static_cols becomes empty or only has item_id
        if not current_static_cols: # Should not happen if item_id_col is always there
            static_features_df_item = pd.DataFrame({"item_id": [history_df_ag["item_id"].iloc[0]]})
        elif "item_id" not in current_static_cols and "item_id" in history_df_ag.columns: # Add item_id if missing from static list but present
            current_static_cols.append("item_id")
            static_features_df_item = history_df_ag[list(set(current_static_cols))].groupby("item_id").first().reset_index()
        elif "item_id" in current_static_cols:
             static_features_df_item = history_df_ag[current_static_cols].groupby("item_id").first().reset_index()
        else: # Fallback if item_id is somehow missing everywhere
            print("Warning: item_id column not found for static features generation during predict.")
            static_features_df_item = pd.DataFrame()


        ts_history = TimeSeriesDataFrame.from_data_frame(
            history_df_ag,
            id_column="item_id",
            timestamp_column="timestamp",
            static_features_df=static_features_df_item
        )

        # 2. Prepare future known covariates
        # CORRECTED LINE: Removed prediction_length argument
        future_skeleton_df = self.predictor.make_future_data_frame(
            data=ts_history
        )
        
        fut_df_with_dyn_features = generate_future_known_features(
            future_skeleton_df.reset_index(), 
            known_dynamic_feature_names=self.known_dynamic_features
        )

        if hasattr(self.preprocessor, 'dynamic_scaler') and self.preprocessor.dynamic_scaler.mean_ is not None and self.known_dynamic_features:
            cols_to_scale = [col for col in self.known_dynamic_features if col in fut_df_with_dyn_features.columns]
            
            if cols_to_scale:
                scaler_expected_features = self.preprocessor.dynamic_cols
                temp_df_for_scaling = pd.DataFrame(0.0, index=fut_df_with_dyn_features.index, columns=scaler_expected_features)
                
                for col in cols_to_scale:
                    if col in temp_df_for_scaling.columns: 
                        temp_df_for_scaling[col] = fut_df_with_dyn_features[col]
                
                scaled_values = self.preprocessor.dynamic_scaler.transform(temp_df_for_scaling[scaler_expected_features])
                scaled_df = pd.DataFrame(scaled_values, columns=scaler_expected_features, index=fut_df_with_dyn_features.index)

                for col in cols_to_scale: 
                    if col in scaled_df.columns:
                       fut_df_with_dyn_features[col] = scaled_df[col]
        
        known_covariates_tsdf = future_skeleton_df.copy()
        # Ensure all known dynamic features columns exist in known_covariates_tsdf before assignment
        for col in self.known_dynamic_features:
            if col not in known_covariates_tsdf.columns:
                known_covariates_tsdf[col] = 0 # Or np.nan, depending on how model handles missing covariates
            if col in fut_df_with_dyn_features.columns:
                # Align indices just in case, though they should be identical
                known_covariates_tsdf[col] = fut_df_with_dyn_features.set_index(known_covariates_tsdf.index)[col]


        # 3. Predict
        predictions_tsdf = self.predictor.predict(ts_history, known_covariates=known_covariates_tsdf)
        
        # 4. Format output
        if store_id not in predictions_tsdf.index.get_level_values('item_id'):
            print(f"Store ID {store_id} not found in prediction results. Available: {predictions_tsdf.index.get_level_values('item_id').unique()}")
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
            return pd.Series([np.nan] * forecast_horizon_days, index=future_dates, name=self.target_col)

        pred_series_weekly = predictions_tsdf.loc[store_id]['mean'] 

        if pred_series_weekly.empty:
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
            return pd.Series([np.nan] * forecast_horizon_days, index=future_dates, name=self.target_col)

        first_pred_date = pred_series_weekly.index.min()
        actual_forecast_start_date = first_pred_date
        if first_pred_date <= last_known_date: 
            if self.model_freq.startswith("W"): # Assuming weekly frequency, adjust if different
                 actual_forecast_start_date = (last_known_date - pd.Timedelta(days=last_known_date.dayofweek)) + pd.Timedelta(days=7) 
            else: 
                 actual_forecast_start_date = last_known_date + pd.Timedelta(days=1)
        
        # The daily forecast should align with the requested forecast_horizon_days
        # The weekly predictions from the model cover self.model_prediction_length *weeks*.
        # We need to generate `forecast_horizon_days` daily predictions.

        # Ensure pred_series_weekly is sorted by index before reindexing
        pred_series_weekly = pred_series_weekly.sort_index()

        # Create the target daily index starting from actual_forecast_start_date for forecast_horizon_days
        daily_forecast_index = pd.date_range(
            start=actual_forecast_start_date,
            periods=forecast_horizon_days,
            freq='D'
        )
        
        # Reindex weekly series to daily, forward-filling values
        # This will correctly fill values for each day based on the week's prediction.
        pred_series_daily = pred_series_weekly.reindex(daily_forecast_index, method='ffill')
        
        # If the first prediction date from the model is later than actual_forecast_start_date,
        # ffill might not fill initial values. We might need bfill for those.
        # Or, ensure pred_series_weekly covers the range needed by daily_forecast_index.
        # A robust way: reindex to a wider range then select.
        # However, ffill from the model's actual prediction start should be mostly correct.
        # If pred_series_weekly starts after daily_forecast_index.min(), initial values will be NaN.
        # This can happen if last_known_date is, e.g., a Friday, and model predicts from next Monday.
        # The `actual_forecast_start_date` logic tries to handle this.

        # If after ffill, the first value is still NaN, it means the model's first prediction
        # is after the start of our daily_forecast_index. We can backfill the first valid prediction.
        if pd.isna(pred_series_daily.iloc[0]):
            pred_series_daily = pred_series_daily.bfill()


        pred_series_daily.name = self.target_col
        return pred_series_daily
    
SEQ_LEN, HORIZON = 30, 30
BATCH_SIZE, LR, EPOCHS = 30, 1e-3, 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESSOR_SAVE_PATH = "feature_preprocessor_chronos.joblib"
MODEL_SAVE_DIR = "AutogluonModels_SazeracSales"

df = pd.read_parquet("data/sazerac_sales_prepared.parquet")
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

print("Initial data shape:", df.shape)
store_counts = df.groupby(ITEM_ID_COL).size()
valid_stores = store_counts[store_counts >= 70].index
df = df[df[ITEM_ID_COL].isin(valid_stores)]
print(f"Shape after filtering stores with < 70 entries: {df.shape}, {len(valid_stores)} stores remaining.")

df[TARGET_COL_NAME] = np.where(df[TARGET_COL_NAME] < 0, 0, df[TARGET_COL_NAME])
q999 = df[TARGET_COL_NAME].quantile(0.999)
df[TARGET_COL_NAME] = np.where(df[TARGET_COL_NAME] > q999, q999, df[TARGET_COL_NAME])
print("Target variable cleaned (negatives to 0, capped at 99.9th percentile).")

store_ids = df[ITEM_ID_COL].unique()
np.random.seed(42)
np.random.shuffle(store_ids)

n = len(store_ids)
n_train = int(0.7 * n)

train_item_ids = store_ids[:n_train]

test_item_ids  = store_ids[n_train:] # Remainder for test

train_df_orig = df[df[ITEM_ID_COL].isin(train_item_ids)]

test_df_orig  = df[df[ITEM_ID_COL].isin(test_item_ids)]
print(f"Data split: Train stores: {len(train_item_ids)}, Test stores: {len(test_item_ids)}")

preprocessor = FeaturePreprocessorChronos(
    static_cat_cols=STATIC_CATEGORICAL,
    static_num_cols=STATIC_NUM,
    dynamic_cols=DYNAMIC,
    target_cols=TARGET
)
print("Fitting preprocessor on training data...")
preprocessor.fit(train_df_orig)
print(f"Preprocessor fitted. Saving to {PREPROCESSOR_SAVE_PATH}")
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)

train_df_ag = train_df_orig.rename(columns={
    ITEM_ID_COL: "item_id",
    TIMESTAMP_COL: "timestamp",
    TARGET_COL_NAME: TARGET_COL_NAME
})

df_transformed_for_static = preprocessor.transform(df.copy())
df_transformed_for_static = df_transformed_for_static.rename(columns={ITEM_ID_COL: "item_id", TIMESTAMP_COL: "timestamp"})


static_features_all_items = df_transformed_for_static[
    ["item_id"] + STATIC_CATEGORICAL + STATIC_NUM
].groupby("item_id").first().reset_index()

# Create train TimeSeriesDataFrame
train_data_tsdf = TimeSeriesDataFrame.from_data_frame(
    train_df_ag,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_all_items[static_features_all_items["item_id"].isin(train_df_ag["item_id"].unique())]
)

train_df_transformed = preprocessor.transform(train_df_orig.copy())
train_df_ag_transformed = train_df_transformed.rename(columns={
    ITEM_ID_COL: "item_id",
    TIMESTAMP_COL: "timestamp",
    TARGET_COL_NAME: TARGET_COL_NAME 
})

train_data_tsdf = TimeSeriesDataFrame.from_data_frame(
    train_df_ag_transformed,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_all_items[static_features_all_items["item_id"].isin(train_df_ag_transformed["item_id"].unique())]
)
print("Training TimeSeriesDataFrame prepared.")
print("Train data columns for AG:", train_data_tsdf.columns)
if not all(kdf in train_data_tsdf.columns for kdf in KNOWN_DYNAMIC_FEATURES):
    missing_kdfs = [kdf for kdf in KNOWN_DYNAMIC_FEATURES if kdf not in train_data_tsdf.columns]
    print(f"WARNING: Missing known dynamic features in training data: {missing_kdfs}")

prediction_length = 8
eval_metric = "MASE" 

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=TARGET_COL_NAME,
    known_covariates_names=KNOWN_DYNAMIC_FEATURES,
    freq="W",
    eval_metric=eval_metric,
    path=MODEL_SAVE_DIR,
)

print("Starting AutoGluon model training...")
predictor.fit(
    train_data_tsdf,
    presets="medium_quality",
    hyperparameters={
         "Chronos": {
            "model_path": "amazon/chronos-t5-small",
            "fine_tune": True,
            "fine_tune_epochs": 5,
            "batch_size": 64,
            "target_scaler": "standard",
            "context_length": prediction_length * 4,
         }
    },
    time_limit=3600,
    enable_ensemble=False,
)
print("AutoGluon training complete.")
print(f"Model saved to: {predictor.path}")
print(f"Preprocessor saved to: {PREPROCESSOR_SAVE_PATH}")