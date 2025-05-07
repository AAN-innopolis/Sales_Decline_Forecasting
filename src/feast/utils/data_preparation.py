"""
Утилиты для подготовки данных и их сохранения в формате, совместимом с Feast.
"""
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Union, List

def save_features_for_feast(
    df: pd.DataFrame,
    feature_type: str,
    output_dir: str = "data/prepared",
    timestamp_column: str = "date",
    entity_column: str = "store_id",
) -> str:
    """
    Сохраняет DataFrame с признаками в формате, совместимом с Feast (Parquet).
    
    Args:
        df: DataFrame с признаками
        feature_type: Тип признаков ('base', 'lstm', 'tft', 'llm')
        output_dir: Директория для сохранения файлов
        timestamp_column: Имя столбца с временной меткой
        entity_column: Имя столбца с идентификатором сущности
        
    Returns:
        Путь к сохраненному файлу
    """
    # Создаем директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем копию DataFrame, чтобы не изменять оригинал
    df_feast = df.copy()
    
    # Преобразуем столбец с временной меткой, если нужно
    if timestamp_column in df_feast.columns and not pd.api.types.is_datetime64_any_dtype(df_feast[timestamp_column]):
        df_feast[timestamp_column] = pd.to_datetime(df_feast[timestamp_column])
    
    # Переименовываем колонки для Feast
    if timestamp_column != "event_timestamp":
        df_feast = df_feast.rename(columns={timestamp_column: "event_timestamp"})
    
    if entity_column != "store":
        df_feast = df_feast.rename(columns={entity_column: "store"})
    
    # Добавляем created_timestamp, если его нет
    if "created_timestamp" not in df_feast.columns:
        df_feast["created_timestamp"] = datetime.now()
    
    # Определяем путь для сохранения
    output_path = os.path.join(output_dir, f"{feature_type}_features.parquet")
    
    # Сохраняем в формате Parquet
    df_feast.to_parquet(output_path, index=False)
    
    print(f"Сохранено {len(df_feast)} строк в {output_path}")
    return output_path 