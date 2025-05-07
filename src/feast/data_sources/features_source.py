"""
Определение источников данных для Feast.
"""

from feast import FileSource
from feast.types import Float32, Int32, Int64

# Источник базовых признаков
base_features_source = FileSource(
    name="base_features_source",
    path="data/prepared/base_features.parquet",
    timestamp_field="date",
    description="Источник базовых признаков для прогнозирования продаж"
)

# Источник LSTM-признаков
lstm_features_source = FileSource(
    name="lstm_features_source",
    path="data/prepared/lstm_features.parquet",
    timestamp_field="date",
    description="Источник LSTM-признаков для прогнозирования продаж"
)

# Источник TFT-признаков
tft_features_source = FileSource(
    name="tft_features_source",
    path="data/prepared/tft_features.parquet",
    timestamp_field="date",
    description="Источник TFT-признаков для прогнозирования продаж"
)

# Источник LLM-признаков
llm_features_source = FileSource(
    name="llm_features_source",
    path="data/prepared/llm_features.parquet",
    timestamp_field="date",
    description="Источник LLM-признаков для прогнозирования продаж"
) 