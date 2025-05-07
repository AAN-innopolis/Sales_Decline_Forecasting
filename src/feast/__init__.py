"""
Модуль Feast для хранения функций и компонентов, связанных с Feature Store.
"""

from utils import (
    save_features_for_feast,
    get_feature_store,
    materialize_features,
    get_online_features
)

from feature_views import (
    base_features_view, 
    lstm_features_view, 
    tft_features_view, 
    llm_features_view
)
from entities import (
    store_entity
)

from data_sources import (
    base_features_source,
    lstm_features_source,
    tft_features_source,
    llm_features_source
)


FEAST_REGISTRY_OBJECTS = [
    store_entity,
    base_features_view,
    lstm_features_view,
    tft_features_view,
    llm_features_view
]

__all__ = [
    'save_features_for_feast',
    'get_feature_store',
    'materialize_features',
    'get_online_features',
    'base_features_view',
    'lstm_features_view',
    'tft_features_view',
    'llm_features_view',
    'store_entity',
    'base_features_source',
    'lstm_features_source',
    'tft_features_source',
    'llm_features_source'
] 