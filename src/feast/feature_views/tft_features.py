"""
Определение представления TFT-признаков для Feast.
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int32

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import store_entity
from data_sources import tft_features_source

tft_features_view = FeatureView(
    name="tft_features_view",
    entities=[store_entity],
    ttl=timedelta(days=365 * 2),  # 2 года
    schema=[
        Field(name="scaled_sales", dtype=Float32),
        Field(name="trend_component", dtype=Float32),
        Field(name="seasonal_component", dtype=Float32),
        Field(name="residual_component", dtype=Float32),
        Field(name="sale_dollars_lag_7", dtype=Float32),
        Field(name="sale_dollars_lag_14", dtype=Float32),
        Field(name="sale_dollars_lag_28", dtype=Float32),
        Field(name="sale_dollars_roll_30_mean", dtype=Float32),
        Field(name="sale_dollars_roll_30_std", dtype=Float32),
        Field(name="sale_dollars_roll_7_mean", dtype=Float32),
        Field(name="sale_dollars_roll_90_mean", dtype=Float32),
        Field(name="transaction_count_roll_30_mean", dtype=Float32),
        Field(name="seasonality_strength", dtype=Float32),
        Field(name="trend_strength", dtype=Float32),
        Field(name="forecast_horizon", dtype=Int32),
    ],
    source=tft_features_source,
    online=True,
    tags={"category": "tft", "version": "1.0"},
    description="TFT-специфичные признаки для прогнозирования продаж",
)