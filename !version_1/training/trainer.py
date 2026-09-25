import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import math

from .losses import VesselLoss
from .metrics import compute_metrics


class Trainer:
    """Trainer для обучения модели"""

    def __init__(
            self,
            model: nn.Module,
            config,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Функция потерь (Custom loss)
        self.criterion = VesselLoss(
            weights={'mse': 1.0, 'direction': 0.1, 'physics': 0.0}
        )

        # Scheduler (уменьшение LR при плато)
        # Убрали параметр verbose, который deprecated в новых версиях PyTorch
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Состояние
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_rmses = []
        self.learning_rates = []

    def train(self):
        """Основной цикл обучения"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(1, self.config.num_epochs + 1):
            # 1. Обучение
            train_metrics = self.train_epoch(epoch)

            # 2. Валидация
            val_metrics = self.evaluate(self.val_loader)

            # Сохраняем историю
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_maes.append(val_metrics['mae'])
            self.val_rmses.append(val_metrics['rmse'])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # 3. Логирование
            self._log_epoch(epoch, train_metrics, val_metrics)

            # 4. Scheduler step
            val_loss = val_metrics['loss']
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # Выводим сообщение, если LR изменился
            if old_lr != new_lr:
                print(f"  ReduceLROnPlateau: reducing learning rate to {new_lr:.6f}")

            # 5. Checkpointing и Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print("  ✓ New best model saved!")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config.early_stopping_patience}")

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}", unit="batch")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            # target передается для Teacher Forcing
            predictions = self.model(batch_x, target=batch_y)

            # Loss calculation
            loss_dict = self.criterion(predictions, batch_y)
            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return {'loss': avg_loss}

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Валидация / Тестирование"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass (без teacher forcing на валидации)
                predictions = self.model(batch_x, target=None, teacher_forcing_ratio=0.0)

                # Loss
                loss_dict = self.criterion(predictions, batch_y)
                total_loss += loss_dict['total_loss'].item()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Агрегация результатов
        avg_loss = total_loss / len(loader)

        # Вычисление метрик (MAE, RMSE, R2)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss

        return metrics

    def _log_epoch(self, epoch, train_metrics, val_metrics):
        """Вывод результатов в консоль"""
        print(f"\nEpoch {epoch}/{self.config.num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def save_checkpoint(self, filename: str):
        """Сохранение модели"""
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, path / filename)

    def load_checkpoint(self, filepath: str):
        """Загрузка модели"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"  ✓ Model loaded from {filepath}")