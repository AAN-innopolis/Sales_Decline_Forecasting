"""
Определение представления LSTM-признаков для Feast.
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int32

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import store_entity
from data_sources import lstm_features_source

lstm_features_view = FeatureView(
    name="lstm_features_view",
    entities=[store_entity],
    ttl=timedelta(days=365 * 2),  # 2 года
    schema=[
        Field(name="normalized_sales", dtype=Float32),
        Field(name="sales_diff", dtype=Float32),
        Field(name="sales_momentum", dtype=Float32),
        Field(name="prev_purchase_sale_dollars", dtype=Float32),
        Field(name="hist_mean_purchases_sale_dollars", dtype=Float32),
        Field(name="hist_std_purchases_sale_dollars", dtype=Float32),
        Field(name="purchase_momentum_30", dtype=Float32),
        Field(name="purchase_momentum_pct_30", dtype=Float32),
        Field(name="days_since_prev_purchase", dtype=Int32),
        Field(name="sin_day", dtype=Float32),
        Field(name="cos_day", dtype=Float32),
        Field(name="sin_week", dtype=Float32),
        Field(name="cos_week", dtype=Float32),
        Field(name="sin_month", dtype=Float32),
        Field(name="cos_month", dtype=Float32),
        Field(name="sin_year", dtype=Float32),
        Field(name="cos_year", dtype=Float32),
    ],
    source=lstm_features_source,
    online=True,
    tags={"category": "lstm", "version": "1.0"},
    description="LSTM-специфичные признаки для прогнозирования продаж",
) 