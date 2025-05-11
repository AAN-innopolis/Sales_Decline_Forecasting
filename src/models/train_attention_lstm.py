"""
Script for training LSTM model with attention mechanism.
"""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.serialization import add_safe_globals
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_utils import setup_logger
from src.config.configs import settings
from src.models.attention_lstm import HybridLSTMAttn

# Добавляем безопасные глобальные объекты для загрузки
add_safe_globals([
    'torch.utils.data.dataloader.DataLoader',
    'torch._utils._rebuild_tensor_v2',
    'torch.storage._TypedStorage',
    'collections.OrderedDict'
])

def custom_collate(batch):
    """
    Пользовательская функция для сборки батча, которая корректно обрабатывает временные метки.
    
    Args:
        batch: Список элементов батча
        
    Returns:
        Словарь с собранными данными батча
    """
    features = []
    targets = []
    timestamps = []
    
    for item in batch:
        # Преобразуем временные метки в числовой формат (например, timestamp в секундах)
        if isinstance(item.get('timestamp'), pd.Timestamp):
            timestamps.append(item['timestamp'].timestamp())
        else:
            timestamps.append(item.get('timestamp', 0))
            
        # Собираем признаки и целевые переменные
        if isinstance(item['features'], torch.Tensor):
            features.append(item['features'].detach().clone())
        else:
            features.append(torch.tensor(item['features'], dtype=torch.float32))
            
        if isinstance(item['target'], torch.Tensor):
            targets.append(item['target'].detach().clone())
        else:
            targets.append(torch.tensor(item['target'], dtype=torch.float32))
    
    # Преобразуем списки в тензоры
    features = torch.stack(features)
    targets = torch.stack(targets)
    timestamps = torch.tensor(timestamps, dtype=torch.float32)
    
    # Проверяем и корректируем размерности
    if len(features.shape) == 3 and features.shape[1] == 1:
        features = features.squeeze(1)  # Убираем лишнюю размерность если она есть
    
    if len(targets.shape) == 2 and targets.shape[1] == 1:
        targets = targets.repeat(1, 5)  # Повторяем значение для каждого шага прогноза
    
    return {
        'features': features,  # (batch_size, input_dim)
        'target': targets,    # (batch_size, prediction_length)
        'timestamp': timestamps
    }

def calculate_metrics(y_true, y_pred):
    """
    Вычисляет метрики качества прогноза.
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
        
    Returns:
        dict: Словарь с метриками
    """
    # Проверяем размерности
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
    
    # Вычисляем метрики
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'mae': mae,
        'rmse': rmse
    }

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    model_dir: str,
    logger: logging.Logger,
    max_grad_norm: float = 1.0  # Максимальная норма градиента
) -> tuple:
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        patience: Early stopping patience
        model_dir: Directory to save model
        logger: Logger instance
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (training history, best metrics)
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_metrics = {
        'val_loss': float('inf'),
        'val_mae': float('inf'),
        'val_rmse': float('inf'),
        'epoch': 0
    }
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        # Training phase
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            attention_mask = torch.ones(features.size(0), features.size(1)).to(device)
            
            optimizer.zero_grad()
            outputs = model(features, attention_mask)
            loss = criterion(outputs, targets)
            
            # Проверяем loss на NaN
            if torch.isnan(loss):
                logger.warning("NaN loss detected! Skipping batch...")
                continue
                
            loss.backward()
            
            # Обрезаем градиенты для предотвращения взрыва
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Проверяем, есть ли валидные потери
        if train_losses:
            epoch_train_loss = np.mean(train_losses)
            history['train_loss'].append(epoch_train_loss)
            logger.info(f" → Train Loss: {epoch_train_loss:.4f}")
        else:
            logger.warning("No valid losses in this epoch!")
            continue
        
        # Validation phase
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for batch in val_bar:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                attention_mask = torch.ones(features.size(0), features.size(1)).to(device)
                
                outputs = model(features, attention_mask)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
                val_bar.set_postfix(val_loss=f"{loss.item():.4f}")
        
        if val_losses:
            epoch_val_loss = np.mean(val_losses)
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Вычисляем метрики
            metrics = calculate_metrics(all_targets, all_preds)
            val_mae = metrics['mae']
            val_rmse = metrics['rmse']
            
            history['val_loss'].append(epoch_val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            
            logger.info(f" → Val Loss: {epoch_val_loss:.4f}")
            logger.info(f" → Val MAE: {val_mae:.4f}")
            logger.info(f" → Val RMSE: {val_rmse:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                }, model_dir / 'best_model.pth')
                
                best_metrics = {
                    'val_loss': epoch_val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'epoch': epoch
                }
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                break
        else:
            logger.warning("No valid validation losses in this epoch!")
            continue
    
    logger.info("\n=== Training Finished ===")
    logger.info(f"Best epoch: {best_metrics['epoch']}")
    logger.info(f"Best validation loss: {best_metrics['val_loss']:.4f}")
    logger.info(f"Best validation MAE: {best_metrics['val_mae']:.4f}")
    logger.info(f"Best validation RMSE: {best_metrics['val_rmse']:.4f}")
    
    return history, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model with attention')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Length of input sequences')
    parser.add_argument('--prediction-length', type=int, default=5,  # Изменено на 5 в соответствии с DAG
                        help='Length of prediction horizon')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,  # Уменьшаем скорость обучения
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    logger = setup_logger(name=__name__, level=args.log_level)
    
    try:
        # Load pre-split datasets with weights_only=False for backward compatibility
        datasets_dir = Path(settings.PROJECT_ROOT, 'data/prepared/lstm_datasets')
        
        # Создаем новые DataLoader'ы с пользовательской функцией сборки батча
        train_dataset = torch.load(datasets_dir / 'train_loader.pt', weights_only=False).dataset
        val_dataset = torch.load(datasets_dir / 'val_loader.pt', weights_only=False).dataset
        test_dataset = torch.load(datasets_dir / 'test_loader.pt', weights_only=False).dataset
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=4
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=4
        )
        
        logger.info("Pre-split datasets loaded successfully")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Get number of features from the first batch
        sample_batch = next(iter(train_loader))
        num_features = sample_batch['features'].shape[-1]
        logger.info(f"Input features shape: {sample_batch['features'].shape}")
        logger.info(f"Target shape: {sample_batch['target'].shape}")
        
        # Create model with correct dimensions
        model = HybridLSTMAttn(
            input_dim=num_features,
            seq_len=sample_batch['features'].shape[1],  # Используем фактическую длину последовательности
            lstm_hidden=128,
            lstm_layers=2,
            dense_hidden=100,
            forecast_horizon=args.prediction_length,
            dropout=0.1,
            num_heads=4
        ).to(device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.L1Loss()
        
        # Train model
        history, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.patience,
            model_dir=Path(settings.PROJECT_ROOT, 'models/attention_lstm'),
            logger=logger,
            max_grad_norm=1.0  # Добавляем ограничение градиентов
        )
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main() 