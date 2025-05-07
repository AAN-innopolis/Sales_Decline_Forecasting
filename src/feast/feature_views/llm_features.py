"""
Определение представления LLM-признаков для Feast.
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, String

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import store_entity
from data_sources import llm_features_source

llm_features_view = FeatureView(
    name="llm_features_view",
    entities=[store_entity],
    ttl=timedelta(days=365 * 2),  # 2 года
    schema=[
        Field(name="sales_trend_description", dtype=String),
        Field(name="store_performance_summary", dtype=String),
        Field(name="sales_anomaly_score", dtype=Float32),
        Field(name="sales_seasonality_strength", dtype=Float32),
        Field(name="sales_forecast_confidence", dtype=Float32),
        Field(name="sentiment_score", dtype=Float32),
        Field(name="store_description", dtype=String),
        Field(name="sales_summary", dtype=String),
        Field(name="llm_input_text", dtype=String),
    ],
    source=llm_features_source,
    online=True,
    tags={"category": "llm", "version": "1.0"},
    description="LLM-специфичные признаки для прогнозирования продаж",
) 