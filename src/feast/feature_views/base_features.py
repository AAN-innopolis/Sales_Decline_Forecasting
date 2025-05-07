"""
Определение представления базовых признаков для Feast.
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int32, Int64
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import store_entity
from data_sources import base_features_source

base_features_view = FeatureView(
    name="base_features_view",
    entities=[store_entity],
    ttl=timedelta(days=365 * 2),  # 2 года
    schema=[
        Field(name="sale_dollars", dtype=Float32),
        Field(name="sale_bottles", dtype=Int32),
        Field(name="sale_liters", dtype=Float32),
        Field(name="transaction_count", dtype=Int32),
        Field(name="unique_categories", dtype=Int32),
        Field(name="unique_items", dtype=Int32),
        Field(name="avg_price_per_bottle", dtype=Float32),
        Field(name="avg_price_per_liter", dtype=Float32),
        Field(name="avg_items_per_transaction", dtype=Float32),
        Field(name="avg_transaction_value", dtype=Float32),
        Field(name="day_of_week_sin", dtype=Float32),
        Field(name="day_of_week_cos", dtype=Float32),
        Field(name="month_sin", dtype=Float32),
        Field(name="month_cos", dtype=Float32),
        Field(name="quarter_sin", dtype=Float32),
        Field(name="quarter_cos", dtype=Float32),
        Field(name="year", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),
    ],
    source=base_features_source,
    online=True,
    tags={"category": "base", "version": "1.0"},
    description="Базовые признаки для прогнозирования продаж",
) 