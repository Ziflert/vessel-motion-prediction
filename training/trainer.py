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
    """Trainer для обучения модели с поддержкой взвешенных целевых переменных"""

    def __init__(
            self,
            model: nn.Module,
            config,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cuda',
            target_scaler=None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # Скалер целей: если задан — evaluate() дополнительно считает метрики
        # в ФИЗИЧЕСКИХ единицах (per-target и per-horizon)
        self.target_scaler = target_scaler

        # Оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Функция потерь с весами из конфига
        self.criterion = VesselLoss(
            target_weights=config.target_weights,
            target_names=config.target_columns,
            loss_weights={'mse': 1.0, 'huber': 0.5, 'smoothness': 0.1}
        )

        # Выводим информацию о весах
        if config.target_weights:
            print("\n📊 Target Weights (for loss calculation):")
            for target, weight in config.target_weights.items():
                print(f"   {target:40s}: {weight:.1f}x")

        # Scheduler
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

        # История per-variable losses
        self.per_variable_history = []

    def train(self):
        """Основной цикл обучения"""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
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

            # Сохраняем per-variable losses
            if 'per_variable_loss' in val_metrics and val_metrics['per_variable_loss']:
                self.per_variable_history.append(val_metrics['per_variable_loss'])

            # 3. Логирование
            self._log_epoch(epoch, train_metrics, val_metrics)

            # 4. Scheduler step
            val_loss = val_metrics['loss']
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

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
        total_mse = 0
        total_huber = 0
        total_smoothness = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}", unit="batch")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
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
            total_mse += loss_dict['mse_loss'].item()
            total_huber += loss_dict['huber_loss'].item()
            total_smoothness += loss_dict['smoothness_loss'].item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{loss_dict['mse_loss'].item():.4f}"
            })

        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'mse_loss': total_mse / n,
            'huber_loss': total_huber / n,
            'smoothness_loss': total_smoothness / n,
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Валидация / Тестирование"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_huber = 0
        total_smoothness = 0
        all_preds = []
        all_targets = []

        # Аккумулятор для per-variable losses
        per_var_accumulator = {}

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass (без teacher forcing)
                predictions = self.model(batch_x, target=None, teacher_forcing_ratio=0.0)

                # Loss
                loss_dict = self.criterion(predictions, batch_y)
                total_loss += loss_dict['total_loss'].item()
                total_mse += loss_dict['mse_loss'].item()
                total_huber += loss_dict['huber_loss'].item()
                total_smoothness += loss_dict['smoothness_loss'].item()

                # Накапливаем per-variable losses
                if loss_dict['per_variable_loss']:
                    for var_name, var_loss in loss_dict['per_variable_loss'].items():
                        if var_name not in per_var_accumulator:
                            per_var_accumulator[var_name] = []
                        per_var_accumulator[var_name].append(var_loss)

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Усредняем
        n = len(loader)
        avg_loss = total_loss / n
        avg_mse = total_mse / n
        avg_huber = total_huber / n
        avg_smoothness = total_smoothness / n

        # Усредняем per-variable losses
        per_var_losses = {}
        for var_name, losses in per_var_accumulator.items():
            per_var_losses[var_name] = np.mean(losses)

        # Вычисление метрик (MAE, RMSE, R2) в масштабированном пространстве
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss
        metrics['mse_loss'] = avg_mse
        metrics['huber_loss'] = avg_huber
        metrics['smoothness_loss'] = avg_smoothness
        metrics['per_variable_loss'] = per_var_losses

        # Физические метрики (главные для исследования качки)
        if self.target_scaler is not None:
            from .metrics import full_physical_report
            metrics['phys'] = full_physical_report(
                all_preds, all_targets, self.target_scaler, self.config.target_columns
            )

        return metrics

    def _log_epoch(self, epoch, train_metrics, val_metrics):
        """Вывод результатов в консоль"""
        print(f"\nEpoch {epoch}/{self.config.num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(MSE: {train_metrics['mse_loss']:.4f}, "
              f"Huber: {train_metrics['huber_loss']:.4f}, "
              f"Smooth: {train_metrics['smoothness_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Показываем per-variable losses (топ-3 худших)
        if val_metrics.get('per_variable_loss'):
            sorted_vars = sorted(
                val_metrics['per_variable_loss'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            print(f"  Top 3 worst variables:")
            for var_name, var_loss in sorted_vars:
                # Получаем вес этой переменной
                weight = self.config.target_weights.get(var_name, 1.0) if self.config.target_weights else 1.0
                print(f"    {var_name:40s}: {var_loss:.4f} (weight: {weight:.1f}x)")

    def save_checkpoint(self, filename: str):
        """Сохранение модели"""
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'per_variable_history': self.per_variable_history,
        }

        torch.save(checkpoint, path / filename)

    def load_checkpoint(self, filepath: str):
        """Загрузка модели"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

        if 'per_variable_history' in checkpoint:
            self.per_variable_history = checkpoint['per_variable_history']

        print(f"  ✓ Model loaded from {filepath}")