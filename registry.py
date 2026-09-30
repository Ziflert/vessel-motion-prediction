"""
Experiment Registry — система версионирования моделей.

Двухуровневая схема (замена «сырым» папкам v001_v002...):

1. manifest.json — карточка КАЖДОГО запуска, лежит внутри папки версии.
   Пишется ДО обучения (все условия), метрики дописываются после.
   Единственный источник правды при inference (не pickled-конфиг!).

2. experiments.csv — реестр-оглавление всех запусков в корне models_archive/.
   Одна строка на запуск. Читается человеком и pandas-ом.

Принципы:
- run_id = YYYYMMDD-HHMMSS-xxxx: уникален, НИКОГДА не переименовывается;
- папки запусков иммутабельны после завершения;
- статус (candidate/production/archived) обновляется в manifest и реестре;
- inference всегда грузит модель через config_from_manifest (чистый JSON),
  а не через pickle внутри чекпоинта.
"""

import csv
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

MODELS_DIR = Path('models_archive')
REGISTRY_FILE = 'experiments.csv'
MANIFEST_FILE = 'manifest.json'


def _force_utf8_stdio():
    """Windows-консоли часто нужен явный UTF-8 (иначе падает печать R²/эмодзи)."""
    import sys as _sys
    for stream in (_sys.stdout, _sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

REGISTRY_COLUMNS = [
    'run_id', 'created_at', 'status', 'profile', 'n_targets', 'targets',
    'seq_len', 'horizon', 'params', 'best_val_loss',
    'test_mae_phys', 'test_r2_phys', 'test_r2_scaled', 'skill_vs_persistence',
    'data_rows', 'notes', 'path',
]


# ============================================================================
# Manifest
# ============================================================================

def new_run_id(slug: str = '') -> str:
    """run_id формата 20251219-120419-<slug>-a3f2 (дата-время + читаемый слаг + hex).

    slug — короткое имя эксперимента из notes (например 'minimal-E1-K2-s42');
    санитизируется (alnum/-/_), обрезается до 24 символов. hex-суффикс гарантирует
    уникальность. Старые прогоны (без слага) остаются с их run_id.
    """
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = f'{random.randint(0, 0xFFFF):04x}'
    s = re.sub(r'[^A-Za-z0-9_-]+', '-', slug).strip('-').lower()[:24]
    return f'{ts}-{s}-{suffix}' if s else f'{ts}-{suffix}'


def create_run(models_dir: Path, manifest: Dict) -> Path:
    """Создаёт папку запуска и записывает начальный manifest.json."""
    run_dir = Path(models_dir) / manifest['run_id']
    if run_dir.exists():
        raise FileExistsError(f'Run dir already exists: {run_dir}')
    (run_dir / 'checkpoints').mkdir(parents=True)
    (run_dir / 'logs').mkdir()
    write_manifest(run_dir, manifest)
    return run_dir


def write_manifest(run_dir: Path, manifest: Dict):
    with open(Path(run_dir) / MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(run_dir: Path) -> Optional[Dict]:
    path = Path(run_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_manifest(run_dir: Path, updates: Dict):
    """Дописывает/обновляет поля manifest.json (например, результаты)."""
    manifest = load_manifest(run_dir) or {}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
    write_manifest(run_dir, manifest)


# ============================================================================
# Registry CSV
# ============================================================================

def registry_path(models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / REGISTRY_FILE


def append_registry_row(models_dir: Path, row: Dict):
    """Дописывает строку в experiments.csv (создаёт с заголовком при необходимости)."""
    path = registry_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, '') for col in REGISTRY_COLUMNS})


def read_registry(models_dir: Path = MODELS_DIR) -> List[Dict]:
    path = registry_path(models_dir)
    if not path.exists():
        return []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ============================================================================
# Поиск и загрузка моделей
# ============================================================================

def list_runs(models_dir: Path = MODELS_DIR) -> List[Dict]:
    """Все запуски: из manifest.json, с fallback-ом на legacy training_config.json."""
    models_dir = Path(models_dir)
    runs = []
    if not models_dir.exists():
        return runs
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir() or d.name == 'archive':
            # 'archive' — легаси/архивные прогоны: не видны в реестре, онлайне и свипах
            continue
        manifest = load_manifest(d)
        if manifest is not None:
            runs.append({
                'run_id': manifest.get('run_id', d.name),
                'dir': d,
                'manifest': manifest,
                'legacy': False,
            })
            continue
        # Legacy: папки старого формата (v001_20251218_171905)
        cfg_file = d / 'training_config.json'
        if cfg_file.exists():
            with open(cfg_file, 'r', encoding='utf-8') as f:
                legacy_cfg = json.load(f)
            runs.append({
                'run_id': d.name,
                'dir': d,
                'manifest': None,
                'legacy_cfg': legacy_cfg,
                'legacy': True,
            })
    return runs


def resolve_run(ref: str, models_dir: Path = MODELS_DIR) -> Path:
    """
    Находит папку запуска по:
      - точному run_id ('20251219-120419-a3f2') или уникальному префиксу;
      - 'production' — модель со status=production;
      - 'latest' — самый новый по run_id;
      - legacy номеру ('3' → папка v003_*).
    """
    runs = list_runs(models_dir)
    if not runs:
        raise FileNotFoundError(f'No models found in {models_dir}')

    if ref == 'production':
        for r in runs:
            if not r['legacy'] and r['manifest'].get('status') == 'production':
                return r['dir']
        raise FileNotFoundError('No model with status=production')

    if ref == 'latest':
        return sorted(runs, key=lambda r: r['run_id'])[-1]['dir']

    # Точный run_id или уникальный префикс (приоритетнее legacy-номера)
    matches = [r for r in runs if r['run_id'] == ref]
    if not matches:
        matches = [r for r in runs if r['run_id'].startswith(ref)]
    if len(matches) == 1:
        return matches[0]['dir']
    if len(matches) > 1:
        raise ValueError(f'Ambiguous run prefix {ref!r}: '
                         f'{[m["run_id"] for m in matches]}')

    # Legacy номер версии: '3' -> v003_*
    if ref.isdigit():
        prefix = f'v{int(ref):03d}'
        matches = [r for r in runs if r['dir'].name.startswith(prefix)]
        if matches:
            return matches[0]['dir']
        raise FileNotFoundError(f'No legacy model with number {ref}')

    raise FileNotFoundError(f'Run {ref!r} not found')


def delete_run(ref: str, models_dir: Path = MODELS_DIR) -> Path:
    """Удаляет папку запуска и его строку из реестра (по run_id/префиксу/legacy-номеру)."""
    import shutil
    run_dir = resolve_run(ref, models_dir)
    run_id = run_dir.name
    shutil.rmtree(run_dir)
    rebuild_registry(models_dir)
    return run_dir


def rebuild_registry(models_dir: Path = MODELS_DIR):
    """Пересобирает experiments.csv из manifest.json всех папок (CSV — производная от manifests)."""
    rows = []
    for run in list_runs(models_dir):
        m = run['manifest']
        if m is None:
            continue
        results = m.get('results', {})
        scaled = results.get('scaled_test', {})
        phys = results.get('physical', {}).get('overall', {})
        skill = results.get('skill_vs_persistence', {}).get('overall')
        feats = m.get('features', {})
        model = m.get('model', {})
        rows.append({
            'run_id': m.get('run_id', run['run_id']),
            'created_at': m.get('created_at', ''),
            'status': m.get('status', ''),
            'profile': feats.get('profile', ''),
            'n_targets': len(feats.get('targets', [])),
            'targets': ';'.join(feats.get('targets', [])),
            'seq_len': model.get('sequence_length', ''),
            'horizon': model.get('prediction_horizon', ''),
            'params': results.get('model', {}).get('total_parameters', ''),
            'best_val_loss': results.get('best_val_loss', ''),
            'test_mae_phys': f"{phys['mae']:.4f}" if phys.get('mae') is not None else '',
            'test_r2_phys': f"{phys['r2']:.4f}" if phys.get('r2') is not None else '',
            'test_r2_scaled': f"{scaled['r2']:.4f}" if scaled.get('r2') is not None else '',
            'skill_vs_persistence': f"{skill:.3f}" if isinstance(skill, (int, float)) else '',
            'data_rows': m.get('data', {}).get('rows_total', ''),
            'notes': m.get('hypothesis', '') or m.get('legacy', {}).get('internal_version_name', ''),
            'path': str(run['dir']),
        })
    rows.sort(key=lambda r: r['run_id'])
    path = registry_path(models_dir)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_trained_model(run_dir: Path, map_location: str = 'cpu'):
    """
    Универсальный загрузчик обученной модели по папке запуска.

    Returns:
        (model, config, feature_scaler, target_scaler, manifest)
        model — в режиме eval на CPU (или указанном устройстве).
    """
    import pickle
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    from models.vessel_predictor import VesselPredictor

    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(f'manifest.json not found in {run_dir} '
                                '(запустите scripts/migrate_models_archive.py для legacy-папок)')

    config = config_from_manifest(manifest)

    # Скалеры: цели — из manifest (mean/std), признаки — из pickle
    scalers_path = run_dir / 'checkpoints' / 'scalers.pkl'
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    ts_params = manifest.get('scalers', {})
    target_scaler = scalers['targets']
    if ts_params.get('target_mean'):
        target_scaler = StandardScaler()
        target_scaler.mean_ = np.asarray(ts_params['target_mean'], dtype=np.float64)
        target_scaler.scale_ = np.asarray(ts_params['target_std'], dtype=np.float64)
        target_scaler.var_ = target_scaler.scale_ ** 2
        target_scaler.n_features_in_ = len(target_scaler.mean_)

    checkpoint = torch.load(run_dir / 'checkpoints' / 'best_model.pt',
                            map_location=map_location, weights_only=False)
    model = VesselPredictor(input_dim=config.input_dim,
                            output_dim=config.output_dim, config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, scalers['features'], target_scaler, manifest


def config_from_manifest(manifest: Dict):
    """
    Восстанавливает Config из manifest.json (БЕЗ pickle).
    Используется при inference — гарантирует правильные признаки/цели/архитектуру.

    BUG-LSTM-07: восстанавливаются ВСЕ вычислительно значимые поля
    (prediction_step, loss_weights, weight_decay, batch_size, max_grad_norm,
    teacher_forcing_ratio, профиль) — для старых manifest берутся дефолты,
    для новых — записанные при обучении значения (round-trip идентичность запуска).
    """
    from config.config import Config

    features = manifest['features']
    model = manifest.get('model', {})
    training = manifest.get('training', {})

    config = Config(
        profile='custom',
        feature_columns=list(features['input']),
        target_columns=list(features['targets']),
        target_weights=dict(features.get('target_weights') or {}),
        cyclic_encoding=False,        # список признаков в manifest уже итоговый
        relative_wave_angle=False,
        verbose=False,
    )

    config.sequence_length = int(model.get('sequence_length', 120))
    config.prediction_horizon = int(model.get('prediction_horizon', 20))
    if 'prediction_step' in model:
        config.prediction_step = int(model['prediction_step'])
    config.encoder_hidden_dims = list(model.get('encoder_hidden_dims', [128, 96]))
    config.temporal_hidden_size = int(model.get('temporal_hidden_size', 128))
    config.temporal_num_layers = int(model.get('temporal_num_layers', 2))
    config.decoder_hidden_dim = int(model.get('decoder_hidden_dim', 96))
    config.bidirectional = bool(model.get('bidirectional', False))
    config.use_attention = bool(model.get('use_attention', True))
    if 'encoder_dropout' in model:
        config.encoder_dropout = float(model['encoder_dropout'])
    if 'temporal_dropout' in model:
        config.temporal_dropout = float(model['temporal_dropout'])
    if 'decoder_dropout' in model:
        config.decoder_dropout = float(model['decoder_dropout'])
    # Профиль — информационно (признаки уже заданы; не перезапускает __post_init__)
    if features.get('profile'):
        config.profile = str(features['profile'])
    # Обучаемые параметры: round-trip идентичность запуска (BUG-LSTM-07)
    if 'batch_size' in training:
        config.batch_size = int(training['batch_size'])
    if 'learning_rate' in training:
        config.learning_rate = float(training['learning_rate'])
    if 'weight_decay' in training:
        config.weight_decay = float(training['weight_decay'])
    if 'max_grad_norm' in training:
        config.max_grad_norm = float(training['max_grad_norm'])
    if 'loss_weights' in training:
        config.loss_weights = dict(training['loss_weights'])
    if 'teacher_forcing_ratio' in training:
        config.teacher_forcing_ratio = float(training['teacher_forcing_ratio'])
    config.device = 'cpu'

    return config


# ============================================================================
# Служебное: воспроизводимость
# ============================================================================

def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> Optional[str]:
    """Хэш текущего коммита (None если не git-репозиторий)."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        return None


def save_pip_freeze(run_dir: Path):
    try:
        out = subprocess.check_output(
            ['pip', 'freeze'], stderr=subprocess.DEVNULL, timeout=60
        ).decode()
        (Path(run_dir) / 'environment.txt').write_text(out, encoding='utf-8')
    except Exception:
        pass
