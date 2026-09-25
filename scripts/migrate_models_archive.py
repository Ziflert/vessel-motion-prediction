"""
Однократная миграция legacy-папок models_archive/ (v001...v004) в новую систему
версионирования: для каждой папки создаётся manifest.json, и все модели
добавляются в experiments.csv.

Запуск:  python scripts/migrate_models_archive.py
Повторный запуск безопасен: папки с уже существующим manifest.json пропускаются.

Примечание о весах целей: в legacy training_config.json веса не сохранялись
(они жили только в pickled-конфиге внутри чекпоинта). Веса восстанавливаются
из определений профилей config/config.py и помечаются в manifest как
"reconstructed" — на inference они не влияют (нужны только при обучении).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg  # noqa: E402

# Веса для 5-целевого motion-профиля ранних версий (реконструкция по
# историческому профилю motion_prediction без Velocity.Vertical)
LEGACY_5T_WEIGHTS = {
    'Pitch(градусы)': 1.0,
    'Roll(градусы)': 2.0,
    'Vertical(Метр)': 1.5,
    'Velocity.Pitching(°/мин)': 0.8,
    'Velocity.Rolling(°/мин)': 1.2,
}

# Точные веса full_prediction (из config/config.py)
FULL_8T_WEIGHTS = {
    'Pitch(градусы)': 1.0,
    'Roll(градусы)': 2.0,
    'Vertical(Метр)': 1.5,
    'Velocity.Pitching(°/мин)': 0.8,
    'Velocity.Rolling(°/мин)': 1.2,
    'Velocity.Vertical(узлы)': 0.8,
    'ROT(°/мин)': 1.5,
    'SOG(узлы)': 1.0,
}

# Внутренние имена версий (из training_summary.txt) для notes
INTERNAL_NAMES = {
    'v001_20251218_171905': 'v003',
    'v002_20251218_220910': 'v004',
    'v003_20251218_221330': 'v007 (best, R2=0.62 scaled)',
    'v004_20251219_120419': 'v004 (full_prediction)',
}

NOTES = {
    'v001_20251218_171905': 'Legacy-модель. Внутр. имя v003. Первый прогон (CPU), horizon=30.',
    'v002_20251218_220910': 'Legacy-модель. Внутр. имя v004. horizon=15.',
    'v003_20251218_221330': 'Legacy-модель. Внутр. имя v007 — ЛУЧШАЯ по R2 (scaled). horizon=10. '
                            'Папка была переименована вручную из v007 в v003.',
    'v004_20251219_120419': 'Legacy-модель. full_prediction, 8 целей, 474K параметров, horizon=20.',
}


def reconstruct_weights(targets):
    tset = set(targets)
    if tset == set(FULL_8T_WEIGHTS):
        return dict(FULL_8T_WEIGHTS), 'exact_from_full_prediction_profile'
    if tset <= set(LEGACY_5T_WEIGHTS):
        return {t: LEGACY_5T_WEIGHTS.get(t, 1.0) for t in targets}, 'reconstructed_from_legacy_motion_profile'
    return {t: 1.0 for t in targets}, 'unknown_equal_weights'


def migrate():
    models_dir = PROJECT_ROOT / 'models_archive'
    migrated, skipped = [], []

    for version_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if (version_dir / reg.MANIFEST_FILE).exists():
            skipped.append(version_dir.name)
            continue

        cfg_file = version_dir / 'training_config.json'
        metrics_file = version_dir / 'training_metrics.json'
        if not cfg_file.exists():
            print(f'!! {version_dir.name}: no training_config.json, skipping')
            continue

        with open(cfg_file, 'r', encoding='utf-8') as f:
            legacy_cfg = json.load(f)
        legacy_metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                legacy_metrics = json.load(f)

        model = legacy_cfg.get('model', {})
        training = legacy_cfg.get('training', {})
        features = legacy_cfg.get('features', {})
        data = legacy_cfg.get('data', {})
        targets = features.get('target_features', [])
        weights, weights_source = reconstruct_weights(targets)

        test_scaled = legacy_metrics.get('test_metrics', {})
        training_meta = legacy_metrics.get('training', {})
        model_meta = legacy_metrics.get('model', {})

        manifest = {
            'run_id': version_dir.name,          # legacy id = имя папки, стабильно
            'created_at': legacy_cfg.get('datetime', ''),
            'status': 'archived',
            'hypothesis': NOTES.get(version_dir.name, 'Legacy model'),
            'git_commit': None,
            'legacy': {
                'internal_version_name': INTERNAL_NAMES.get(version_dir.name, 'unknown'),
                'original_version_number': legacy_cfg.get('version'),
                'migrated': True,
            },
            'data': {
                'rows_total': data.get('total_rows'),
                'rows_train': data.get('train_rows'),
                'rows_val': data.get('val_rows'),
                'rows_test': data.get('test_rows'),
                'split': {'scheme': 'chunked_concatenated (legacy, windows could cross chunk borders)'},
                'path': 'data/raw/your_data.csv',
                'sha256': None,
            },
            'model': {
                'type': model.get('type', 'lstm'),
                'sequence_length': model.get('sequence_length'),
                'prediction_horizon': model.get('prediction_horizon'),
                'encoder_hidden_dims': model.get('encoder_hidden_dims'),
                'temporal_hidden_size': model.get('temporal_hidden_size'),
                'temporal_num_layers': model.get('temporal_num_layers'),
                'decoder_hidden_dim': model.get('decoder_hidden_dim'),
                'bidirectional': model.get('bidirectional', False),
                'use_attention': model.get('use_attention', True),
            },
            'training': training,
            'features': {
                'profile': 'legacy (5-target motion-like)' if len(targets) == 5 else 'full_prediction',
                'input': features.get('input_features', []),
                'targets': targets,
                'target_weights': weights,
                'target_weights_source': weights_source,
                'feature_engineering': None,   # старые модели обучены на сырых углах
                'input_dim': model.get('input_dim'),
                'output_dim': model.get('output_dim'),
            },
            'scalers': {
                'type': 'StandardScaler',
                'file': 'checkpoints/scalers.pkl',
                'note': 'mean/std not extracted during migration (no torch env); '
                        'target params will be taken from pickle at inference',
            },
            'results': {
                'total_epochs': training_meta.get('total_epochs'),
                'training_time_seconds': training_meta.get('training_time_seconds'),
                'best_val_loss': training_meta.get('best_validation_loss'),
                'scaled_test': {
                    'loss': test_scaled.get('loss'),
                    'mae': test_scaled.get('mae'),
                    'rmse': test_scaled.get('rmse'),
                    'r2': test_scaled.get('r2'),
                },
                'model': model_meta,
                'note': 'Метрики — в СТАНДАРТИЗОВАННОМ пространстве (см. PROJECT_OVERVIEW.md). '
                        'Физические метрики — в results/inference_*/summary_statistics.txt.',
            },
            'updated_at': 'migrated',
        }

        reg.write_manifest(version_dir, manifest)

        reg.append_registry_row(models_dir, {
            'run_id': version_dir.name,
            'created_at': manifest['created_at'],
            'status': 'archived',
            'profile': manifest['features']['profile'],
            'n_targets': len(targets),
            'targets': ';'.join(targets),
            'seq_len': model.get('sequence_length', ''),
            'horizon': model.get('prediction_horizon', ''),
            'params': model_meta.get('total_parameters', ''),
            'best_val_loss': training_meta.get('best_validation_loss', ''),
            'test_mae_phys': '',  # неизвестно для legacy (см. results/inference_*)
            'test_r2_phys': '',
            'test_r2_scaled': test_scaled.get('r2', ''),
            'skill_vs_persistence': '',
            'data_rows': data.get('total_rows', ''),
            'notes': NOTES.get(version_dir.name, ''),
            'path': str(version_dir),
        })
        migrated.append(version_dir.name)

    print(f'Migrated: {migrated or "none"}')
    print(f'Skipped (already have manifest): {skipped or "none"}')


if __name__ == '__main__':
    migrate()
