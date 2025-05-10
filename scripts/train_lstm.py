#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для обучения LSTM модели прогнозирования продаж на n дней вперед.
Считывает датасет в формате parquet, подготавливает данные и обучает модель для прогнозирования временных рядов.
"""

import os
import sys
import argparse
import logging
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Настройка логирования
def setup_logger(log_level=logging.INFO, log_file=None):
    """Настройка логгера с указанным уровнем логирования."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, 
                       handlers=[logging.StreamHandler()])
    
    logger = logging.getLogger(__name__)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

# Класс для подготовки данных в формате последовательностей
class StoreTimeSeriesDataset(Dataset):
    """
    Класс для подготовки последовательностей данных для LSTM модели прогнозирования временных рядов.
    Включает характеристики магазина вместо его ID.
    """
    def __init__(self, X, store_features, y, seq_length=7, forecast_horizon=1):
        """
        Инициализация датасета.
        
        Args:
            X (numpy.ndarray): Матрица временных признаков
            store_features (numpy.ndarray): Матрица характеристик магазинов
            y (numpy.ndarray): Целевые значения продаж
            seq_length (int): Длина входной последовательности для LSTM
            forecast_horizon (int): Горизонт прогнозирования (на сколько дней вперед)
        """
        self.X = X
        self.store_features = store_features
        self.y = y
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        """Возвращает количество доступных последовательностей."""
        return max(0, len(self.X) - self.seq_length - self.forecast_horizon + 1)
    
    def __getitem__(self, idx):
        """
        Возвращает последовательность признаков, характеристики магазина и целевое значение продаж.
        
        Args:
            idx (int): Индекс последовательности
            
        Returns:
            tuple: (последовательность признаков, характеристики магазина, целевое значение продаж)
        """
        # Получаем последовательность признаков
        X_seq = self.X[idx:idx+self.seq_length]
        # Характеристики магазина для целевого дня
        store_feats = self.store_features[idx+self.seq_length]
        # Целевое значение продаж на forecast_horizon дней вперед
        y_target = self.y[idx+self.seq_length:idx+self.seq_length+self.forecast_horizon]
        
        return X_seq, store_feats, y_target

# Модель LSTM с учетом характеристик магазина для прогнозирования временных рядов
class StoreLSTMModel(nn.Module):
    """
    LSTM модель для прогнозирования продаж на n дней вперед с учетом характеристик магазина.
    Вместо эмбеддингов использует характеристики магазина, что позволяет
    делать прогнозы для ранее невиданных магазинов.
    """
    def __init__(self, time_features_size, store_features_size, hidden_size, forecast_horizon=1, num_layers=2, dropout=0.2):
        """
        Инициализация модели.
        
        Args:
            time_features_size (int): Размерность временных признаков
            store_features_size (int): Размерность характеристик магазина
            hidden_size (int): Размерность скрытого состояния LSTM
            forecast_horizon (int): Горизонт прогнозирования (на сколько дней вперед)
            num_layers (int): Количество слоев LSTM
            dropout (float): Вероятность dropout
        """
        super(StoreLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM слои для временных признаков
        self.lstm = nn.LSTM(
            input_size=time_features_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Слой для обработки характеристик магазина
        self.store_fc = nn.Sequential(
            nn.Linear(store_features_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Полносвязные слои для объединения временных и магазинных признаков
        self.fc1 = nn.Linear(hidden_size + 64, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, forecast_horizon)  # Выходной слой предсказывает forecast_horizon значений
        
    def forward(self, x, store_features):
        """
        Прямой проход модели.
        
        Args:
            x (torch.Tensor): Входные последовательности временных признаков [batch_size, seq_len, time_features_size]
            store_features (torch.Tensor): Характеристики магазинов [batch_size, store_features_size]
            
        Returns:
            torch.Tensor: Прогноз продаж на forecast_horizon дней вперед [batch_size, forecast_horizon]
        """
        # Формируем начальные скрытые состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прогоняем временные признаки через LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]  # Берем выход с последнего временного шага
        
        # Обработка характеристик магазина
        store_out = self.store_fc(store_features)
        
        # Объединяем выход LSTM и характеристики магазина
        combined = torch.cat((lstm_out, store_out), dim=1)
        
        # Проходим через полносвязные слои
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Линейный выход для регрессии
        
        return out

def preprocess_data(df, target_column, seq_length, forecast_horizon, logger, min_history_length=30):
    """
    Предобработка данных для LSTM модели прогнозирования временных рядов.
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм
        target_column (str): Название целевой колонки (продажи)
        seq_length (int): Длина последовательности для LSTM
        forecast_horizon (int): Горизонт прогнозирования (сколько дней вперед)
        logger (logging.Logger): Логгер
        min_history_length (int): Минимальное количество записей для магазина, чтобы использовать его для обучения
        
    Returns:
        tuple: (датасет для обучения, датасет для тестирования, скейлеры признаков)
    """
    logger.info("Начинаем предобработку данных")
    
    # Проверяем наличие целевой колонки
    if target_column not in df.columns:
        logger.error(f"Целевая колонка {target_column} не найдена в данных")
        raise ValueError(f"Целевая колонка {target_column} не найдена в данных")
    
    # Фильтруем магазины с недостаточной историей данных
    store_counts = df['store'].value_counts()
    valid_stores = store_counts[store_counts >= min_history_length + forecast_horizon].index.tolist()
    
    # Логируем информацию о фильтрации
    total_stores = len(store_counts)
    filtered_stores = total_stores - len(valid_stores)
    logger.info(f"Всего магазинов в данных: {total_stores}")
    logger.info(f"Отфильтровано магазинов с историей менее {min_history_length + forecast_horizon} записей: {filtered_stores} ({filtered_stores/total_stores*100:.2f}%)")
    logger.info(f"Оставлено магазинов для обучения: {len(valid_stores)}")
    
    # Фильтруем датафрейм, оставляя только магазины с достаточной историей
    df_filtered = df[df['store'].isin(valid_stores)].copy()
    logger.info(f"Размер данных после фильтрации: {len(df_filtered)} строк (было {len(df)} строк)")
    
    # Если после фильтрации данных стало слишком мало, предупреждаем об этом
    if len(df_filtered) < len(df) * 0.5:
        logger.warning("После фильтрации осталось менее 50% данных. Возможно, стоит уменьшить min_history_length.")
    
    # Используем отфильтрованный датафрейм для дальнейшей обработки
    df = df_filtered
    
    # Разделяем признаки на временные и характеристики магазинов
    # Характеристики магазинов - это признаки, которые не меняются во времени для одного магазина
    # или меняются редко (агрегаты по магазину)
    store_feature_cols = [
        'store_avg_sales', 'store_size', 'city_avg_sales', 'county_avg_sales',
        'store_transaction_count', 'store_to_city_sales_ratio', 'store_to_county_sales_ratio',
        'category_avg_sales', 'bottle_volume_avg_sales', 'pack_avg_sales',
        'category_to_total_sales_ratio', 'lon', 'lat'
    ]
    
    # Проверяем наличие признаков магазина в данных
    available_store_features = [col for col in store_feature_cols if col in df.columns]
    
    if not available_store_features:
        logger.warning("Не найдены признаки магазинов. Используем только временные признаки.")
        # Если нет признаков магазинов, добавляем хотя бы один фиктивный признак
        df['store_dummy'] = 1.0
        available_store_features = ['store_dummy']
    
    # Исключаем категориальные и идентификационные колонки, а также признаки магазинов из временных
    exclude_cols = [
        'invoice_line_no', 'name', 'address', 'city', 'zipcode', 'county',
        'category', 'category_name', 'itemno', 'im_desc', 'holiday_name',
        'sale_dollars_decrease_day', 'sale_dollars_decrease_week_avg',
        'sale_dollars_significant_decrease', 'sale_dollars_3day_consecutive_decrease',
        'store'  # Исключаем ID магазина из временных признаков
    ] + available_store_features
    
    # Оставляем только числовые временные признаки
    time_feature_cols = [col for col in df.columns if col not in exclude_cols 
                         and col != target_column and df[col].dtype != 'object']
    
    logger.info(f"Используем {len(time_feature_cols)} временных признаков и {len(available_store_features)} признаков магазинов")
    logger.info(f"Временные признаки: {time_feature_cols}")
    logger.info(f"Признаки магазинов: {available_store_features}")
    
    # Проверяем на пропущенные значения
    missing_time = df[time_feature_cols].isna().sum().sum()
    missing_store = df[available_store_features].isna().sum().sum()
    missing_target = df[target_column].isna().sum()
    
    if missing_time > 0:
        logger.warning(f"Найдено {missing_time} пропущенных значений в временных признаках. Заполняем нулями.")
        df[time_feature_cols] = df[time_feature_cols].fillna(0)
    
    if missing_store > 0:
        logger.warning(f"Найдено {missing_store} пропущенных значений в признаках магазинов. Заполняем нулями.")
        df[available_store_features] = df[available_store_features].fillna(0)
    
    if missing_target > 0:
        logger.warning(f"Найдено {missing_target} пропущенных значений в целевой переменной. Заполняем средними значениями.")
        df[target_column] = df[target_column].fillna(df[target_column].mean())
    
    # Нормализация признаков
    time_scaler = StandardScaler()
    store_scaler = StandardScaler()
    target_scaler = StandardScaler()  # Для нормализации целевой переменной
    
    X_time_scaled = time_scaler.fit_transform(df[time_feature_cols])
    X_store_scaled = store_scaler.fit_transform(df[available_store_features])
    y_scaled = target_scaler.fit_transform(df[[target_column]]).flatten()  # Нормализуем целевую переменную
    
    logger.info("Разделение на обучающую и тестовую выборки...")
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df['store']
    )
    logger.info("Использована стратификация по магазинам")
    
    X_time_train, X_time_test = X_time_scaled[train_idx], X_time_scaled[test_idx]
    X_store_train, X_store_test = X_store_scaled[train_idx], X_store_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    
    # Создаем наборы данных для PyTorch
    train_dataset = StoreTimeSeriesDataset(X_time_train, X_store_train, y_train, 
                                          seq_length=seq_length, forecast_horizon=forecast_horizon)
    test_dataset = StoreTimeSeriesDataset(X_time_test, X_store_test, y_test, 
                                         seq_length=seq_length, forecast_horizon=forecast_horizon)
    
    logger.info(f"Размер обучающей выборки: {len(train_dataset)}")
    logger.info(f"Размер тестовой выборки: {len(test_dataset)}")
    
    return train_dataset, test_dataset, (time_scaler, store_scaler, target_scaler), (time_feature_cols, available_store_features)

def train_model(model, train_loader, criterion, optimizer, device, epochs, logger, patience=5):
    """
    Обучение модели с ранней остановкой.
    
    Args:
        model (nn.Module): Модель для обучения
        train_loader (DataLoader): Загрузчик данных для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        device (torch.device): Устройство для вычислений
        epochs (int): Максимальное количество эпох
        logger (logging.Logger): Логгер
        patience (int): Количество эпох для ранней остановки
        
    Returns:
        list: История потерь по эпохам
    """
    logger.info(f"Начинаем обучение модели на {device}")
    model.train()
    losses = []
    
    best_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Используем tqdm для отображения прогресса
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1}/{epochs}')
        
        for X_batch, store_feats_batch, y_batch in progress_bar:
            # Перемещаем тензоры на устройство
            X_batch = X_batch.float().to(device)
            store_feats_batch = store_feats_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(X_batch, store_feats_batch)
            
            # Вычисление потерь
            loss = criterion(outputs, y_batch)
            
            # Обратное распространение
            loss.backward()
            
            # Оптимизация
            optimizer.step()
            
            # Накапливаем потери
            epoch_loss += loss.item()
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Средняя потеря за эпоху
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        logger.info(f'Эпоха [{epoch+1}/{epochs}], Потери: {avg_loss:.6f}')
        
        # Проверка для ранней остановки
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            logger.info(f"Ранняя остановка на эпохе {epoch+1}")
            break
    
    return losses

def evaluate_model(model, test_loader, criterion, device, logger, target_scaler=None):
    """
    Оценка модели на тестовой выборке.
    
    Args:
        model (nn.Module): Модель для оценки
        test_loader (DataLoader): Загрузчик данных для тестирования
        criterion: Функция потерь
        device (torch.device): Устройство для вычислений
        logger (logging.Logger): Логгер
        target_scaler: Скейлер для обратного преобразования целевой переменной
        
    Returns:
        dict: Метрики модели
    """
    logger.info("Оценка модели на тестовой выборке")
    model.eval()
    test_loss = 0.0
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for X_batch, store_feats_batch, y_batch in tqdm(test_loader, desc="Оценка"):
            X_batch = X_batch.float().to(device)
            store_feats_batch = store_feats_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            
            # Прямой проход
            outputs = model(X_batch, store_feats_batch)
            
            # Потери
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            # Сохраняем истинные и предсказанные значения
            y_true_all.extend(y_batch.cpu().numpy())
            y_pred_all.extend(outputs.cpu().numpy())
    
    # Преобразуем списки в массивы
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Вычисление метрик для каждого шага прогноза
    forecast_horizon = y_true_all.shape[1] if len(y_true_all.shape) > 1 else 1
    
    metrics = {}
    avg_loss = test_loss / len(test_loader)
    metrics['loss'] = avg_loss
    
    # Возвращаем значения в исходный масштаб, если предоставлен скейлер
    if target_scaler is not None:
        # Преобразуем формат для обратного масштабирования
        y_true_reshaped = y_true_all.reshape(-1, 1) if forecast_horizon == 1 else y_true_all.reshape(-1, 1)
        y_pred_reshaped = y_pred_all.reshape(-1, 1) if forecast_horizon == 1 else y_pred_all.reshape(-1, 1)
        
        # Обратное масштабирование
        y_true_orig = target_scaler.inverse_transform(y_true_reshaped).reshape(y_true_all.shape)
        y_pred_orig = target_scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_all.shape)
    else:
        y_true_orig = y_true_all
        y_pred_orig = y_pred_all
    
    # Рассчитываем метрики для каждого шага прогноза
    for h in range(forecast_horizon):
        if forecast_horizon > 1:
            y_true_h = y_true_orig[:, h]
            y_pred_h = y_pred_orig[:, h]
        else:
            y_true_h = y_true_orig
            y_pred_h = y_pred_orig
        
        mse = mean_squared_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_h, y_pred_h)
        r2 = r2_score(y_true_h, y_pred_h)
        
        step_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        if forecast_horizon > 1:
            metrics[f'horizon_{h+1}'] = step_metrics
            logger.info(f"Метрики для горизонта {h+1}:")
        else:
            metrics.update(step_metrics)
        
        logger.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Общая MSE, RMSE, MAE для всех горизонтов
    total_mse = mean_squared_error(y_true_orig.flatten(), y_pred_orig.flatten())
    total_rmse = np.sqrt(total_mse)
    total_mae = mean_absolute_error(y_true_orig.flatten(), y_pred_orig.flatten())
    
    logger.info(f"Общие метрики для всех горизонтов:")
    logger.info(f"Тестовая потеря: {avg_loss:.6f}")
    logger.info(f"MSE: {total_mse:.4f}, RMSE: {total_rmse:.4f}, MAE: {total_mae:.4f}")
    
    metrics['total_mse'] = total_mse
    metrics['total_rmse'] = total_rmse
    metrics['total_mae'] = total_mae
    metrics['y_true'] = y_true_orig
    metrics['y_pred'] = y_pred_orig
    
    return metrics

def prepare_store_data_for_prediction(df, store_id, feature_cols, scalers, seq_length=7):
    """
    Подготовка данных магазина для прогнозирования.
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм
        store_id (int): ID магазина
        feature_cols (tuple): Кортеж списков признаков (time_feature_cols, store_feature_cols)
        scalers (tuple): Кортеж скейлеров (time_scaler, store_scaler, target_scaler)
        seq_length (int): Требуемая длина последовательности
        
    Returns:
        tuple: (Временные признаки, Характеристики магазина)
    """
    time_feature_cols, store_feature_cols = feature_cols
    time_scaler, store_scaler, _ = scalers
    
    # Фильтруем данные для выбранного магазина
    store_df = df[df['store'] == store_id].copy()
    
    # Проверяем, что у магазина есть данные
    if len(store_df) == 0:
        raise ValueError(f"Магазин с ID {store_id} не найден в данных")
    
    # Проверяем, достаточно ли данных для формирования последовательности
    if len(store_df) < seq_length:
        raise ValueError(f"У магазина {store_id} недостаточно данных ({len(store_df)} < {seq_length}). Магазин не будет рассмотрен.")
    
    # Проверяем наличие всех необходимых признаков
    missing_time_cols = [col for col in time_feature_cols if col not in store_df.columns]
    missing_store_cols = [col for col in store_feature_cols if col not in store_df.columns]
    
    if missing_time_cols or missing_store_cols:
        missing_cols = missing_time_cols + missing_store_cols
        raise ValueError(f"Отсутствуют необходимые признаки для магазина {store_id}: {missing_cols}")
    
    # Сортируем по индексу (дате), чтобы данные были в правильном временном порядке
    store_df = store_df.sort_index()
    
    # Заполняем пропущенные значения нулями
    store_df_time = store_df[time_feature_cols].fillna(0)
    store_df_store = store_df[store_feature_cols].fillna(0)
    
    # Получаем временные признаки в том же порядке, что и при обучении
    time_features = time_scaler.transform(store_df_time)
    
    # Получаем признаки магазина (берем первую строку, так как они должны быть одинаковы для магазина)
    store_features = store_scaler.transform(store_df_store.iloc[[0]])
    
    return time_features, store_features[0]

def predict_store_future(model, store_features, initial_time_features, scalers, seq_length, forecast_days, device):
    """
    Прогнозирование продаж для конкретного магазина на будущие периоды.
    
    Args:
        model (nn.Module): Обученная модель
        store_features (numpy.ndarray): Характеристики магазина
        initial_time_features (numpy.ndarray): Начальные временные признаки для прогнозирования
        scalers (tuple): Кортеж скейлеров (time_scaler, store_scaler, target_scaler)
        seq_length (int): Длина последовательности для LSTM
        forecast_days (int): Количество дней для прогнозирования
        device (torch.device): Устройство для вычислений
        
    Returns:
        numpy.ndarray: Массив прогнозов продаж
    """
    _, _, target_scaler = scalers
    model.eval()
    
    # Начальная последовательность временных признаков
    # Берем последние seq_length записей
    if len(initial_time_features) < seq_length:
        raise ValueError(f"Недостаточно данных для формирования последовательности. Требуется {seq_length}, доступно {len(initial_time_features)}")
    
    current_sequence = initial_time_features[-seq_length:].copy()
    
    # Проверяем размерность входных данных
    expected_features = model.lstm.input_size
    actual_features = current_sequence.shape[1]
    
    if expected_features != actual_features:
        raise ValueError(f"Несоответствие размерности признаков в начальной последовательности. Ожидается: {expected_features}, Получено: {actual_features}")
    
    # Массив для хранения прогнозов
    predictions = []
    
    # Нормализованные характеристики магазина
    store_features_scaled = store_features.reshape(1, -1)
    
    with torch.no_grad():
        # Определяем размер горизонта прогнозирования модели
        model_forecast_horizon = model.forecast_horizon
        
        # Количество итераций зависит от требуемого количества дней и горизонта модели
        iterations = (forecast_days + model_forecast_horizon - 1) // model_forecast_horizon
        
        for i in range(iterations):
            # Проверяем размерность последовательности перед каждым прогнозом
            if current_sequence.shape[1] != expected_features:
                print(f"Итерация {i}: Неправильная размерность последовательности: {current_sequence.shape}")
                # Исправляем размерность, если нужно (заполняем нулями)
                fixed_sequence = np.zeros((seq_length, expected_features))
                # Копируем имеющиеся признаки
                for j in range(seq_length):
                    features_to_copy = min(current_sequence.shape[1], expected_features)
                    fixed_sequence[j, :features_to_copy] = current_sequence[j, :features_to_copy]
                current_sequence = fixed_sequence
            
            # Последовательность для прогноза
            X = torch.FloatTensor(current_sequence.reshape(1, seq_length, -1)).to(device)
            
            # Проверяем размерность тензора
            if X.size(2) != expected_features:
                raise ValueError(f"Итерация {i}: Несоответствие размерности тензора. Ожидается: {expected_features}, Получено: {X.size(2)}")
                
            store_feats = torch.FloatTensor(store_features_scaled).to(device)
            
            # Получаем прогноз на модельный горизонт
            output = model(X, store_feats)
            
            # Преобразуем результат в numpy
            forecast = output.cpu().numpy().flatten()
            
            # Если используем скейлер целевой переменной, инвертируем масштабирование
            if target_scaler is not None:
                forecast = target_scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            
            # Добавляем только нужное количество прогнозов, чтобы не превысить forecast_days
            remaining = forecast_days - len(predictions)
            forecasts_to_add = min(remaining, len(forecast))
            predictions.extend(forecast[:forecasts_to_add])
            
            # Если предсказали достаточно дней, заканчиваем
            if len(predictions) >= forecast_days:
                break
            
            # В реальной системе здесь должна быть логика обновления последовательности
            # с учетом новых данных. Для простоты примера сдвигаем последовательность
            # и добавляем прогноз в конец
            next_features = current_sequence[-1].copy()  # Последние известные признаки
            current_sequence = np.vstack([current_sequence[1:], next_features.reshape(1, -1)])
    
    return np.array(predictions)

def plot_store_predictions(df, predictions_dict, stores_info, target_column, output_dir, logger):
    """
    Построение графиков временных рядов продаж по рассмотренным магазинам.
    Особое внимание уделяется магазинам с уменьшающимися продажами.
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм
        predictions_dict (dict): Словарь с прогнозами по магазинам
        stores_info (pd.DataFrame): Информация о магазинах
        target_column (str): Название целевой колонки (продажи)
        output_dir (str): Директория для сохранения результатов
        logger (logging.Logger): Логгер
    """
    if not predictions_dict:
        logger.warning("Нет данных прогнозов для построения графиков")
        return
    
    logger.info("Построение графиков временных рядов по магазинам")
    
    # Создаем директорию для графиков магазинов
    store_plots_dir = os.path.join(output_dir, 'store_plots')
    os.makedirs(store_plots_dir, exist_ok=True)
    
    # Анализируем тренды для всех магазинов
    trend_data = {}
    for store_id, predictions in predictions_dict.items():
        x = np.arange(len(predictions))
        slope = np.polyfit(x, predictions, 1)[0]
        avg_prediction = np.mean(predictions)
        trend_data[store_id] = {"slope": slope, "avg_prediction": avg_prediction}
    
    # Сортируем магазины по среднему прогнозу (по возрастанию)
    sorted_stores = sorted(trend_data.keys(), key=lambda x: trend_data[x]["avg_prediction"])
    
    # Получаем список магазинов с падающим трендом
    declining_stores = [store_id for store_id in trend_data if trend_data[store_id]["slope"] < 0]
    
    # Сначала обрабатываем магазины с наименьшими продажами и падающим трендом
    priority_stores = list(set(sorted_stores[:5]).union(set(declining_stores[:5])))
    remaining_stores = [store_id for store_id in predictions_dict if store_id not in priority_stores]
    
    # Объединяем списки в нужном порядке
    ordered_stores = priority_stores + remaining_stores
    
    for store_id in ordered_stores:
        try:
            predictions = predictions_dict[store_id]
            
            # Получаем данные магазина
            store_df = df[df['store'] == store_id].copy()
            
            if len(store_df) == 0:
                logger.warning(f"Нет данных для построения графика магазина {store_id}")
                continue
            
            # Определяем тренд
            slope = trend_data[store_id]["slope"]
            trend_label = "Падение" if slope < 0 else "Рост" if slope > 0 else "Стабильно"
            trend_color = 'red' if slope < 0 else 'green' if slope > 0 else 'blue'
            
            # Создаем фигуру для графика
            plt.figure(figsize=(14, 8))
            
            # Получаем фактические даты из индекса
            historical_dates = store_df.index
            
            # Создаем даты для прогноза, продолжая от последней даты исторических данных
            last_date = historical_dates[-1]
            
            # Определяем шаг между датами на основе имеющихся данных
            if len(historical_dates) > 1:
                date_diff = historical_dates[1] - historical_dates[0]
            else:
                # Если только одна дата, предполагаем, что данные дневные
                date_diff = pd.Timedelta(days=1)
            
            # Создаем даты прогноза
            forecast_dates = [last_date + (i+1)*date_diff for i in range(len(predictions))]
            
            # Построение исторических данных
            plt.plot(historical_dates, store_df[target_column], label='Исторические данные', color='blue')
            
            # Последнее историческое значение
            last_historical = store_df[target_column].iloc[-1] if len(store_df) > 0 else 0
            
            # Построение прогноза
            plt.plot(forecast_dates, predictions, label=f'Прогноз (тренд: {trend_label})', 
                     color=trend_color, linestyle='--', linewidth=2)
            
            # Отмечаем начало прогноза
            plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
            plt.text(last_date, last_historical, ' Начало прогноза', 
                     verticalalignment='center', fontsize=10)
            
            # Отмечаем тренд линией регрессии
            x = np.arange(len(predictions))
            trend_line = np.polyval([slope, predictions[0]], x)
            plt.plot(forecast_dates, trend_line, color=trend_color, linestyle=':', 
                     label=f'Линия тренда (наклон: {slope:.4f})')
            
            # Добавляем информацию о магазине
            store_info_text = f"Магазин {store_id}"
            if store_id in stores_info.index:
                store_row = stores_info.loc[store_id]
                city = store_row.get('city', '')
                county = store_row.get('county', '')
                if city and county:
                    store_info_text += f" | {city}, {county}"
                store_avg_sales = store_row.get('store_avg_sales', None)
                if store_avg_sales is not None:
                    store_info_text += f" | Ср. продажи: {store_avg_sales:.2f}"
            
            # Добавляем статистику прогноза
            avg_prediction = np.mean(predictions)
            min_prediction = np.min(predictions)
            max_prediction = np.max(predictions)
            change_percent = ((predictions[-1] - predictions[0]) / predictions[0] * 100) if predictions[0] != 0 else 0
            
            stats_text = (f"Статистика прогноза:\n"
                        f"Среднее: {avg_prediction:.2f}\n"
                        f"Мин: {min_prediction:.2f}\n"
                        f"Макс: {max_prediction:.2f}\n"
                        f"Изменение: {change_percent:.2f}%")
            
            # Добавляем текстовые блоки
            plt.figtext(0.01, 0.97, store_info_text, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.figtext(0.01, 0.90, stats_text, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Заголовок с цветовой индикацией тренда
            title_color = trend_color
            is_priority = store_id in priority_stores
            title_style = {'fontweight': 'bold'} if is_priority else {}
            plt.title(f"Прогноз продаж для {store_info_text}", color=title_color, **title_style)
            
            # Настраиваем формат дат на оси X
            plt.gcf().autofmt_xdate()
            plt.xlabel('Дата')
            plt.ylabel(f'{target_column}')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Добавляем метку если это магазин из приоритетных
            if is_priority:
                priority_note = "ПРИОРИТЕТ: Магазин с " 
                priority_note += "наименьшими продажами" if store_id in sorted_stores[:5] else ""
                if store_id in sorted_stores[:5] and store_id in declining_stores[:5]:
                    priority_note += " и "
                if store_id in declining_stores[:5]:
                    priority_note += "падающим трендом продаж"
                plt.figtext(0.5, 0.01, priority_note, fontsize=12, ha='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
            
            # Сохраняем график
            plot_filename = f"store_{store_id}_forecast"
            if slope < 0:
                plot_filename += "_declining"
            if store_id in sorted_stores[:5]:
                plot_filename += "_lowest"
                
            store_plot_path = os.path.join(store_plots_dir, f"{plot_filename}.png")
            plt.savefig(store_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"График для магазина {store_id} сохранен в {store_plot_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при построении графика для магазина {store_id}: {e}")
    
    logger.info(f"Графики временных рядов сохранены в директории {store_plots_dir}")

def analyze_store_predictions(predictions_dict, stores_info, target_column, output_dir, logger):
    """
    Анализ прогнозов продаж по магазинам и сохранение результатов.
    
    Args:
        predictions_dict (dict): Словарь с прогнозами по магазинам
        stores_info (pd.DataFrame): Информация о магазинах
        target_column (str): Название целевой колонки (продажи)
        output_dir (str): Директория для сохранения результатов
        logger (logging.Logger): Логгер
    """
    logger.info("Анализ прогнозов по магазинам")
    
    # Создаем итоговый датафрейм с прогнозами
    results = []
    
    for store_id, predictions in predictions_dict.items():
        # Рассчитываем метрики прогноза
        avg_prediction = np.mean(predictions)
        min_prediction = np.min(predictions)
        max_prediction = np.max(predictions)
        std_prediction = np.std(predictions)
        
        # Проверяем на тренд (простая линейная регрессия)
        x = np.arange(len(predictions))
        slope = np.polyfit(x, predictions, 1)[0]
        trend = "Рост" if slope > 0 else "Падение" if slope < 0 else "Стабильно"
        
        # Добавляем информацию о магазине
        store_info = {}
        if store_id in stores_info.index:
            store_row = stores_info.loc[store_id]
            store_info = {
                'city': store_row.get('city', ''),
                'county': store_row.get('county', ''),
                'store_avg_sales': float(store_row.get('store_avg_sales', 0)),
                'store_size': float(store_row.get('store_size', 0))
            }
        
        # Конвертируем numpy типы в стандартные Python типы для JSON
        results.append({
            'store_id': int(store_id),
            'avg_prediction': float(avg_prediction),
            'min_prediction': float(min_prediction),
            'max_prediction': float(max_prediction),
            'std_prediction': float(std_prediction),
            'trend': trend,
            'slope': float(slope),
            'predictions': [float(p) for p in predictions],
            **store_info
        })
    
    # Создаем датафрейм результатов
    results_df = pd.DataFrame(results)
    
    # Если нет результатов, логируем предупреждение и возвращаемся
    if len(results_df) == 0:
        logger.warning("Нет результатов прогнозирования для анализа.")
        return
    
    # Создаем отдельный датафрейм со всеми магазинами
    all_results_df = results_df.copy()
    
    # Сортируем по среднему прогнозу (от меньшего к большему)
    results_df = results_df.sort_values('avg_prediction', ascending=True)
    
    # Сохраняем результаты в CSV
    csv_path = os.path.join(output_dir, 'store_predictions.csv')
    save_cols = ['store_id', 'avg_prediction', 'min_prediction', 'max_prediction', 'std_prediction', 'trend']
    if 'city' in results_df.columns:
        save_cols.extend(['city', 'county', 'store_avg_sales', 'store_size'])
    all_results_df[save_cols].to_csv(csv_path, index=False)
    
    # Сохраняем полные прогнозы в JSON
    json_path = os.path.join(output_dir, 'store_predictions_full.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Результаты анализа сохранены в {csv_path} и {json_path}")
    
    # Выводим топ-5 магазинов с наименьшими продажами
    logger.info("\nТоп-5 магазинов с наименьшими прогнозируемыми продажами:")
    for i, row in results_df.head(5).iterrows():
        store_info = f", {row.get('city', '')}, {row.get('county', '')}" if 'city' in row else ""
        logger.info(f"Магазин ID: {row['store_id']}{store_info}, Ср. прогноз: {row['avg_prediction']:.2f}, "
                   f"Тренд: {row['trend']}")
    
    # Дополнительно выводим магазины с падающим трендом
    declining_stores = results_df[results_df['trend'] == 'Падение'].head(5)
    if len(declining_stores) > 0:
        logger.info("\nТоп-5 магазинов с падающим трендом продаж:")
        for i, row in declining_stores.iterrows():
            store_info = f", {row.get('city', '')}, {row.get('county', '')}" if 'city' in row else ""
            logger.info(f"Магазин ID: {row['store_id']}{store_info}, Ср. прогноз: {row['avg_prediction']:.2f}, "
                       f"Наклон тренда: {row['slope']:.4f}")
    
    # Строим гистограмму распределения прогнозов
    plt.figure(figsize=(12, 6))
    plt.hist(results_df['avg_prediction'], bins=20, alpha=0.7, color='blue')
    plt.axvline(results_df['avg_prediction'].mean(), color='red', linestyle='--', 
               label=f'Среднее: {results_df["avg_prediction"].mean():.2f}')
    plt.title(f'Распределение прогнозируемых значений {target_column} по магазинам')
    plt.xlabel(f'Средний прогноз {target_column}')
    plt.ylabel('Количество магазинов')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Сохраняем график
    hist_path = os.path.join(output_dir, 'predictions_distribution.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    logger.info(f"График распределения прогнозов сохранен в {hist_path}")
    
    return results_df

def plot_training_results(loss_history, metrics, output_dir, logger):
    """
    Построение графиков и сохранение результатов обучения.
    
    Args:
        loss_history (list): История потерь по эпохам
        metrics (dict): Словарь с метриками модели
        output_dir (str): Директория для сохранения результатов
        logger (logging.Logger): Логгер
    """
    # Создаем директорию для графиков, если её нет
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # График потерь при обучении
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Динамика потерь при обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_loss.png'), dpi=300)
    
    # График предсказаний vs. истинных значений (для первого горизонта, если их несколько)
    plt.figure(figsize=(10, 6))
    
    # Получаем данные для построения
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    # Если у нас несколько горизонтов прогнозирования, возьмем первый
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
    
    # Берем подмножество для наглядности, если точек много
    max_points = 1000
    if len(y_true) > max_points:
        indices = np.random.choice(len(y_true), max_points, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # Строим диаграмму рассеяния
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    
    # Добавляем диагональную линию (идеальные предсказания)
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Предсказанные vs. Истинные значения')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'predictions_vs_true.png'), dpi=300)
    
    # График остатков (ошибок)
    plt.figure(figsize=(10, 6))
    residuals = y_true_sample - y_pred_sample
    plt.scatter(y_pred_sample, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Остатки')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки (Истинные - Предсказанные)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'residuals.png'), dpi=300)
    
    logger.info(f"Графики результатов сохранены в директории {plots_dir}")

def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Обучение LSTM модели для прогнозирования продаж')
    
    parser.add_argument('--input', type=str, required=True, default='data/sales_features.parquet',
                        help='Путь к входному файлу в формате parquet')
    parser.add_argument('--output', type=str, default='output/lstm_timeseries',
                        help='Директория для сохранения результатов')
    parser.add_argument('--target', type=str, default='sale_dollars',
                        help='Целевая переменная для прогнозирования (продажи)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Уровень логирования')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Путь к файлу для логирования')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Размер батча для обучения')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Максимальное количество эпох обучения')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Скорость обучения')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Размерность скрытого состояния LSTM')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Количество слоев LSTM')
    parser.add_argument('--seq-length', type=int, default=7,
                        help='Длина входной последовательности для LSTM')
    parser.add_argument('--forecast-horizon', type=int, default=30,
                        help='Горизонт прогнозирования (количество дней вперед)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Вероятность dropout')
    parser.add_argument('--forecast-days', type=int, default=180,
                        help='Количество дней для прогнозирования в тестовом режиме')
    parser.add_argument('--stores-to-predict', type=int, default=10,
                        help='Количество магазинов для тестового прогноза')
    parser.add_argument('--min-history-length', type=int, default=30,
                        help='Минимальное количество записей для магазина, чтобы использовать его для обучения')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для воспроизводимости результатов')
    parser.add_argument('--cuda', action='store_true',
                        help='Использовать CUDA, если доступна')
    
    return parser.parse_args()

def main():
    """Основная функция скрипта."""
    # Парсинг аргументов
    args = parse_args()
    
    # Настройка логирования
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_level, args.log_file)
    
    # Выводим значения параметров
    logger.info("Параметры запуска:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    # Устанавливаем seed для воспроизводимости
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Определяем устройство для вычислений
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")
    
    # Создаем директорию для результатов
    os.makedirs(args.output, exist_ok=True)
    
    # Сохраняем параметры запуска
    with open(os.path.join(args.output, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Загружаем данные
    try:
        logger.info(f"Загрузка данных из {args.input}")
        df = pd.read_parquet(args.input).iloc[:100000]  # Ограничиваем для тестирования
        df.sort_index(inplace=True)
        logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        logger.info(f"Период данных: с {df.index.min()} по {df.index.max()}")
        logger.info(f"Колонки в датасете: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # Предобработка данных
    train_dataset, test_dataset, scalers, feature_cols = preprocess_data(
        df, args.target, args.seq_length, args.forecast_horizon, logger, args.min_history_length)
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Получаем размерности признаков
    time_feature_cols, store_feature_cols = feature_cols
    time_features_size = len(time_feature_cols)
    store_features_size = len(store_feature_cols)
    
    # Создаем модель
    model = StoreLSTMModel(
        time_features_size=time_features_size,
        store_features_size=store_features_size,
        hidden_size=args.hidden_size,
        forecast_horizon=args.forecast_horizon,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Создана модель: {model}")
    logger.info(f"Размерность временных признаков: {time_features_size}")
    logger.info(f"Размерность признаков магазина: {store_features_size}")
    
    # Функция потерь и оптимизатор для регрессии
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Засекаем время обучения
    start_time = time.time()
    
    # Обучение модели
    loss_history = train_model(
        model, train_loader, criterion, optimizer, device, args.epochs, logger)
    
    # Время обучения
    training_time = time.time() - start_time
    logger.info(f"Обучение завершено за {training_time:.2f} секунд")
    
    # Оценка модели
    metrics = evaluate_model(model, test_loader, criterion, device, logger, scalers[2])
    
    # Сохраняем модель и все необходимые данные для предсказаний
    model_path = os.path.join(args.output, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'time_features_size': time_features_size,
        'store_features_size': store_features_size,
        'forecast_horizon': args.forecast_horizon
    }, model_path)
    logger.info(f"Модель сохранена в {model_path}")
    
    # Сохраняем списки признаков для последующего использования
    features_path = os.path.join(args.output, 'feature_columns.json')
    with open(features_path, 'w') as f:
        json.dump({
            'time_features': time_feature_cols,
            'store_features': store_feature_cols
        }, f, indent=2)
    logger.info(f"Списки признаков сохранены в {features_path}")
    
    # Строим графики результатов
    plot_training_results(loss_history, metrics, args.output, logger)
    
    # Создаем датафрейм с информацией о магазинах для анализа
    stores_info = df.groupby('store').first().reset_index()
    store_cols = ['store', 'city', 'county', 'store_avg_sales', 'store_size']
    available_cols = [col for col in store_cols if col in stores_info.columns]
    stores_info = stores_info[available_cols].set_index('store')
    
    # Прогнозирование для нескольких магазинов
    logger.info(f"Прогнозирование для {args.stores_to_predict} магазинов на {args.forecast_days} дней")
    
    # Выбираем случайные магазины для прогноза
    all_stores = df['store'].unique()
    store_ids = np.random.choice(all_stores, min(args.stores_to_predict, len(all_stores)), replace=False)
    
    # Прогнозирование для выбранных магазинов
    predictions_dict = {}
    for store_id in store_ids:
        try:
            logger.info(f"Прогнозирование для магазина {store_id}")
            time_features, store_features = prepare_store_data_for_prediction(df, store_id, feature_cols, scalers, args.seq_length)
            
            # Проверяем размерности
            logger.info(f"Размерность временных признаков для магазина {store_id}: {time_features.shape[1]}")
            logger.info(f"Ожидаемая размерность: {time_features_size}")
            
            predictions = predict_store_future(
                model, store_features, time_features, scalers, args.seq_length, args.forecast_days, device)
            
            predictions_dict[store_id] = predictions
            logger.info(f"Успешно выполнен прогноз для магазина {store_id}")
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании для магазина {store_id}: {e}")
    
    # Анализ прогнозов по магазинам (только если есть успешные прогнозы)
    if predictions_dict:
        results_df = analyze_store_predictions(predictions_dict, stores_info, args.target, args.output, logger)
        
        # Построение графиков временных рядов по магазинам
        plot_store_predictions(df, predictions_dict, stores_info, args.target, args.output, logger)
    else:
        logger.warning("Не удалось сделать прогнозы ни для одного магазина.")
    
    logger.info("Работа скрипта завершена успешно")

if __name__ == "__main__":
    main() 