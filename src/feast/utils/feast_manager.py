"""
Утилиты для управления Feast: инициализация, материализация, получение признаков.
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd

from feast import FeatureStore

def get_feature_store(repo_path: Optional[str] = None) -> FeatureStore:
    """
    Получает экземпляр FeatureStore.

    Args:
        repo_path: Путь к репозиторию Feast

    Returns:
        Экземпляр FeatureStore
    """
    if repo_path is None:
        repo_path = os.path.abspath(".")
    
    # Проверяем наличие конфигурационного файла
    config_path = os.path.join(repo_path, "config", "feast", "feature_store.yaml")
    default_path = os.path.join(repo_path, "feature_store.yaml")
    
    fs_yaml_file = config_path if os.path.exists(config_path) else default_path
    
    return FeatureStore(repo_path=repo_path, fs_yaml_file=fs_yaml_file)

def get_online_features(
    entity_id: str,
    repo_path: Optional[str] = None,
    feature_views: Optional[List[str]] = None,
) -> Dict:
    """
    Получает онлайн-признаки для конкретной сущности.

    Args:
        entity_id: Идентификатор сущности (магазина)
        repo_path: Путь к репозиторию Feast
        feature_views: Список представлений признаков для получения

    Returns:
        Словарь с признаками
    """
    print(f"Получение онлайн-признаков для магазина {entity_id}...")
    
    # Получаем экземпляр FeatureStore
    store = get_feature_store(repo_path)
    
    # Если не указаны конкретные представления признаков, используем все
    if feature_views is None:
        feature_views = [
            "base_features_view:*",
            "lstm_features_view:*",
            "tft_features_view:*",
            "llm_features_view:*",
        ]
    
    # Получаем онлайн-признаки
    features = store.get_online_features(
        features=feature_views,
        entity_rows=[{"store": entity_id}],
    ).to_dict()
    
    return features

def materialize_features(
    repo_path: Optional[str] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> None:
    """
    Материализует признаки в онлайн-хранилище.

    Args:
        repo_path: Путь к репозиторию Feast
        start_date: Начальная дата для материализации
        end_date: Конечная дата для материализации
    """
    print("Материализация признаков...")
    
    # Получаем экземпляр FeatureStore
    store = get_feature_store(repo_path)
    
    # Устанавливаем значения по умолчанию, если не указаны
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)  # За последний год
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Материализуем признаки
    store.materialize(start_date=start_date, end_date=end_date)
    
    print(f"Признаки успешно материализованы с {start_date} по {end_date}!") 