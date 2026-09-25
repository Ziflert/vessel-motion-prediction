"""
Smoke-тесты проекта (запускаются БЕЗ torch — с заглушкой).

Покрывают:
  - инженерию признаков (sin/cos, относительный угол волны, согласованность
    списка колонок между функцией для списков и для DataFrame);
  - сегментную нарезку окон (окна не пересекают границы сегментов);
  - физические метрики и skill score;
  - реестр экспериментов (manifest roundtrip, resolve_run, CSV);
  - восстановление Config из manifest (в т.ч. для мигрированной legacy-модели);
  - инженерию признаков в актуальном профиле full_prediction.

Запуск:  py -3.13 tests/test_smoke.py   (или pytest tests/test_smoke.py)
"""

import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Заглушка torch (тесты не требуют реального фреймворка)
# ---------------------------------------------------------------------------
if 'torch' not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        fake_torch = types.ModuleType('torch')
        fake_torch.Tensor = type('Tensor', (), {})  # нужно scipy/sklearn при проверке типов
        fake_torch.is_tensor = lambda x: False
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.version = types.SimpleNamespace(cuda=None)
        fake_torch.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=False))
        fake_torch.device = lambda *a, **k: 'cpu'
        sys.modules['torch'] = fake_torch

        fake_utils = types.ModuleType('torch.utils')
        fake_data = types.ModuleType('torch.utils.data')
        fake_data.Dataset = object
        fake_data.DataLoader = object
        fake_utils.data = fake_data
        sys.modules['torch.utils'] = fake_utils
        sys.modules['torch.utils.data'] = fake_data

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Windows-консоли может не хватать UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Инженерия признаков
# ---------------------------------------------------------------------------

def test_feature_engineering_consistency():
    from data.features import engineer_feature_columns, engineer_features_dataframe

    cols = [
        'Swell(метры)', 'Wave.Highest(метры)', 'Wave.direction(градусы)',
        'Wind.direction(градусы)', 'Rudder State(градусы)',
        'RPM(Обороты в минуту)', 'Pitch(градусы)', 'Roll(градусы)',
        'SOG(узлы)', 'Course(градусы)',
    ]
    eng_cols = engineer_feature_columns(cols, cyclic=True, relative_wave_angle=True)

    assert 'Wave.direction(sin)' in eng_cols and 'Wave.direction(cos)' in eng_cols
    assert 'Wave.direction(градусы)' not in eng_cols, 'исходный циклический угол должен быть заменён'
    assert 'Rudder State(градусы)' in eng_cols, 'угол руля — НЕ циклический, не заменяется'
    assert 'Rel.Wave.angle(sin)' in eng_cols and 'Rel.Wave.angle(cos)' in eng_cols
    assert len(eng_cols) == len(set(eng_cols)), 'дубликаты в списке признаков'

    # DataFrame-версия создаёт ровно те же колонки
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.uniform(0, 360, 60) for c in cols if '(градусы)' in c})
    for c in ['Swell(метры)', 'Wave.Highest(метры)', 'SOG(узлы)', 'RPM(Обороты в минуту)']:
        df[c] = rng.uniform(0, 10, 60)
    df_eng = engineer_features_dataframe(df, cyclic=True, relative_wave_angle=True)

    for c in eng_cols:
        assert c in df_eng.columns, f'колонка {c} не создана в DataFrame'

    # sin/cos совпадают с прямым расчётом
    i = 0
    assert abs(df_eng['Course(sin)'].iloc[i] -
               np.sin(np.deg2rad(df['Course(градусы)'].iloc[i]))) < 1e-9

    # Круговая непрерывность: разность углов 350 -> 10 даёт +20 (а не -340)
    from data.features import circular_diff
    assert abs(circular_diff(350, 10)) == 20.0 or abs(circular_diff(10, 350)) == 20.0
    assert abs(circular_diff(350, 10)) <= 180.0

    print('OK  test_feature_engineering_consistency')


# ---------------------------------------------------------------------------
# 2. Сегментная нарезка окон (фикс утечки)
# ---------------------------------------------------------------------------

def test_dataset_segment_windows():
    from data.dataset import VesselDataset

    class Cfg:
        pass

    cfg = Cfg()
    cfg.sequence_length = 5
    cfg.prediction_horizon = 2
    cfg.prediction_step = 1
    cfg.feature_columns = ['a']
    cfg.target_columns = ['b']

    seg1 = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0) * 2})
    seg2 = pd.DataFrame({'a': np.arange(100.0, 115.0), 'b': np.arange(100.0, 115.0)})

    ds = VesselDataset([seg1, seg2], cfg, fit_scalers=True)

    required = cfg.sequence_length + cfg.prediction_horizon  # 7
    # seg1: 10-7+1=4 окна; seg2: 15-7+1=9 окон
    assert len(ds) == 4 + 9, f'ожидалось 13 окон, получено {len(ds)}'

    # Ни одно окно не пересекает границу сегментов
    for idx in ds.valid_indices:
        start = int(idx)
        seg_of_rows = [0 if start + i < len(seg1) else 1 for i in range(required)]
        assert len(set(seg_of_rows)) == 1, f'окно {start} пересекает границу сегментов!'

    # Обратный тест: на склеенном DataFrame окон было бы больше (утечка/склейка)
    ds_glued = VesselDataset(pd.concat([seg1, seg2], ignore_index=True), cfg, fit_scalers=False,
                             scalers=ds.scalers)
    assert len(ds_glued) > len(ds), 'склейка сегментов должна давать лишние (невалидные) окна'

    print('OK  test_dataset_segment_windows')


# ---------------------------------------------------------------------------
# 3. Физические метрики и skill score
# ---------------------------------------------------------------------------

def _load_module_by_path(name: str, path: Path):
    """Загрузка модуля по пути (мимо package __init__, который тянет torch)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_physical_metrics():
    metrics = _load_module_by_path('metrics_standalone', ROOT / 'training' / 'metrics.py')
    full_physical_report = metrics.full_physical_report
    skill_score = metrics.skill_score

    class FakeScaler:
        """Стандартизация: x_phys = x_scaled * 2 + 10 (std=2, mean=10)."""

        def inverse_transform(self, x):
            return np.asarray(x) * 2.0 + 10.0

    rng = np.random.default_rng(1)
    preds = rng.normal(0, 0.1, size=(32, 4, 2))
    targets = rng.normal(0, 1.0, size=(32, 4, 2))

    report = full_physical_report(preds, targets, FakeScaler(), ['Roll(градусы)', 'Pitch(градусы)'])

    assert set(report['per_target']) == {'Roll(градусы)', 'Pitch(градусы)'}
    assert len(report['per_horizon_mae']) == 4
    assert len(report['per_horizon_per_target_mae']['Roll(градусы)']) == 4
    assert report['overall']['mae'] > 0

    # Физическая MAE = масштабированная * std (в среднем)
    scaled_mae = np.mean(np.abs(preds - targets))
    assert abs(report['overall']['mae'] - scaled_mae * 2.0) < 1e-6

    assert skill_score(0.5, 1.0) == 0.5
    assert skill_score(1.0, 1.0) == 0.0
    assert skill_score(1.5, 1.0) < 0

    print('OK  test_physical_metrics')


# ---------------------------------------------------------------------------
# 4. Реестр экспериментов
# ---------------------------------------------------------------------------

def test_registry():
    import registry as reg

    manifest = {
        'run_id': '20250101-000000-0001',
        'created_at': '2025-01-01T00:00:00',
        'status': 'running',
        'features': {'input': ['a'], 'targets': ['b'], 'target_weights': {'b': 2.0}},
        'model': {'sequence_length': 10, 'prediction_horizon': 3,
                  'encoder_hidden_dims': [8], 'temporal_hidden_size': 8,
                  'temporal_num_layers': 1, 'decoder_hidden_dim': 8,
                  'bidirectional': False, 'use_attention': True},
        'training': {'learning_rate': 0.001},
    }

    with tempfile.TemporaryDirectory() as td:
        md = Path(td)
        run_dir = reg.create_run(md, manifest)
        assert (run_dir / 'manifest.json').exists()
        assert (run_dir / 'checkpoints').exists()

        reg.update_manifest(run_dir, {'status': 'production',
                                      'results': {'scaled_test': {'r2': 0.5}}})
        m2 = reg.load_manifest(run_dir)
        assert m2['status'] == 'production'
        assert m2['results']['scaled_test']['r2'] == 0.5
        assert 'updated_at' in m2

        # resolve_run: production / префикс / latest
        assert reg.resolve_run('production', md) == run_dir
        assert reg.resolve_run('20250101', md) == run_dir
        assert reg.resolve_run('latest', md) == run_dir

        # CSV-реестр
        reg.append_registry_row(md, {'run_id': manifest['run_id'], 'status': 'production'})
        rows = reg.read_registry(md)
        assert rows[0]['run_id'] == manifest['run_id']
        assert rows[0]['status'] == 'production'

    print('OK  test_registry')


# ---------------------------------------------------------------------------
# 5. Восстановление Config из manifest (вкл. мигрированную legacy-модель)
# ---------------------------------------------------------------------------

def test_config_from_manifest():
    from registry import config_from_manifest

    manifest_path = ROOT / 'models_archive' / 'v003_20251218_221330' / 'manifest.json'
    if not manifest_path.exists():
        print('SKIP test_config_from_manifest (миграция не выполнялась)')
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    cfg = config_from_manifest(manifest)
    assert cfg.sequence_length == 120
    assert cfg.prediction_horizon == 10
    assert cfg.input_dim == 17
    assert cfg.output_dim == 5
    assert cfg.feature_columns[0] == 'Pitch(градусы)'
    assert cfg.device == 'cpu'
    assert cfg.target_weights['Roll(градусы)'] == 2.0

    print('OK  test_config_from_manifest')


# ---------------------------------------------------------------------------
# 6. Инженерия в актуальном профиле full_prediction
# ---------------------------------------------------------------------------

def test_config_full_profile_engineering():
    from config.config import Config

    cfg = Config(verbose=False)
    assert cfg.profile == 'full_prediction'
    assert cfg.feature_engineering == {'cyclic_encoding': True, 'relative_wave_angle': True}

    assert 'Wave.direction(sin)' in cfg.feature_columns
    assert 'Wave.direction(градусы)' not in cfg.feature_columns
    assert 'Rel.Wave.angle(sin)' in cfg.feature_columns
    assert 'Rudder State(градусы)' in cfg.feature_columns
    assert 'RPM(Обороты в минуту)' in cfg.feature_columns
    assert 'Course(градусы)' not in cfg.feature_columns          # заменён на sin/cos
    assert 'Course(sin)' in cfg.feature_columns

    # Цели не изменяются инженерией
    assert cfg.target_columns[0] == 'Pitch(градусы)'
    assert len(cfg.target_columns) == 8
    assert cfg.target_weights['Roll(градусы)'] == 2.0

    # motion_core: только позиционные цели
    cfg_core = Config(profile='motion_core_prediction', verbose=False)
    assert cfg_core.target_columns == ['Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)']

    print('OK  test_config_full_profile_engineering')


if __name__ == '__main__':
    test_feature_engineering_consistency()
    test_dataset_segment_windows()
    test_physical_metrics()
    test_registry()
    test_config_from_manifest()
    test_config_full_profile_engineering()
    print('\nВсе smoke-тесты пройдены ✓')
