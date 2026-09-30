"""
Smoke-тесты проекта (запускаются БЕЗ torch — с заглушкой).

Покрывают:
  - инженерию признаков (sin/cos, относительный угол волны, согласованность
    списка колонок между функцией для списков и для DataFrame);
  - сегментную нарезку окон (окна не пересекают границы сегментов);
  - физические метрики и skill score;
  - реестр экспериментов (manifest roundtrip, resolve_run, CSV);
  - восстановление Config из manifest (в т.ч. для мигрированной legacy-модели);
  - инженерию признаков в актуальном профиле full_prediction;
  - regression-тесты аудита 2026-09-29 (BUG-LSTM-01/03/04/07).

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
        # VesselDataset.__getitem__ использует torch.from_numpy (заглушка:
        # возвращаем обычный numpy-массив)
        fake_torch.from_numpy = lambda arr: np.asarray(arr)
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
    assert cfg.feature_engineering == {'cyclic_encoding': True, 'relative_wave_angle': True,
                                       'relative_wind_angle': True}

    assert 'Wave.direction(sin)' in cfg.feature_columns
    assert 'Wave.direction(градусы)' not in cfg.feature_columns
    assert 'Rel.Wave.angle(sin)' in cfg.feature_columns
    assert 'Rel.Wind.angle(sin)' in cfg.feature_columns, \
        'КУ ветра (relative_wind_angle) должен входить в итоговые признаки'
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


# ---------------------------------------------------------------------------
# 7. Regression: BUG-LSTM-01 — relative_wind_angle в online FE
# ---------------------------------------------------------------------------

def test_extract_fe_params_relative_wind():
    from online.sources import extract_fe_params
    from data.features import engineer_feature_columns

    # manifest с явным relative_wind_angle=True: параметр ДОЛЖЕН передаваться
    manifest = {'features': {'feature_engineering': {
        'cyclic_encoding': True, 'relative_wave_angle': True,
        'relative_wind_angle': True}}}
    fe = extract_fe_params(manifest)
    assert fe['relative_wind_angle'] is True, \
        'BUG-LSTM-01: relative_wind_angle потерян при online-инженерии'
    assert fe['cyclic'] is True and fe['relative_wave_angle'] is True

    # Старый manifest без ключа relative_wind_angle: на момент обучения
    # параметр применялся (дефолт True) — fallback True
    fe_old = extract_fe_params({'features': {'feature_engineering': {
        'cyclic_encoding': True, 'relative_wave_angle': True}}})
    assert fe_old['relative_wind_angle'] is True

    # manifest без feature_engineering: без FE (все False)
    fe_none = extract_fe_params({})
    assert fe_none == {'cyclic': False, 'relative_wave_angle': False,
                       'relative_wind_angle': False}

    # Round-trip: список итоговых колонок по fe-параметрам manifest совпадает
    # со списком колонок модели (train = online FE)
    from config.config import Config
    cfg = Config(verbose=False)
    raw_cols, _, _ = Config.get_profile_config(cfg.profile)   # ИСХОДНЫЕ признаки профиля
    eng_cols = engineer_feature_columns(
        raw_cols, cyclic=fe['cyclic'],
        relative_wave_angle=fe['relative_wave_angle'],
        relative_wind_angle=fe['relative_wind_angle'])
    assert eng_cols == cfg.feature_columns, 'train/online FE рассинхронизированы'

    print('OK  test_extract_fe_params_relative_wind')


# ---------------------------------------------------------------------------
# 8. Regression: BUG-LSTM-03 — prediction_step согласован с числом выходов
# ---------------------------------------------------------------------------

def test_dataset_prediction_step():
    from data.dataset import VesselDataset

    class Cfg:
        pass

    cfg = Cfg()
    cfg.sequence_length = 5
    cfg.prediction_horizon = 4
    cfg.prediction_step = 2
    cfg.feature_columns = ['a']
    cfg.target_columns = ['b']

    # Один непрерывный сегмент: 5 + (4-1)*2 + 1 = 12 строк достаточно
    seg = pd.DataFrame({'a': np.arange(12.0), 'b': np.arange(12.0) * 3})
    ds = VesselDataset(seg, cfg, fit_scalers=True)
    assert len(ds) == 1, f'ожидалось 1 окно, получено {len(ds)}'

    x, y = ds[0]
    # Число выходов ВСЕГДА prediction_horizon (а не horizon // step)
    assert y.shape[0] == cfg.prediction_horizon, \
        f'число выходов {y.shape[0]} != prediction_horizon {cfg.prediction_horizon}'
    assert x.shape[0] == cfg.sequence_length
    # Выходные шаги отстоят на prediction_step: для окна с start=0 targets —
    # строки 5, 7, 9, 11 (шаг 2 по времени; сравнение в масштабированном
    # пространстве — скалеры обучены на этом же сегменте)
    expected = ds.scalers['targets'].transform(
        (np.array([5, 7, 9, 11]) * 3.0).reshape(-1, 1))
    assert np.allclose(np.asarray(y), expected), \
        f'выходные шаги не отстоят на prediction_step: {np.asarray(y)}'

    # Сегмент из 13 строк: окна start=0 и start=1 (по 12 строк каждое)
    seg13 = pd.DataFrame({'a': np.arange(13.0), 'b': np.arange(13.0) * 3})
    ds13 = VesselDataset(seg13, cfg, fit_scalers=False, scalers=ds.scalers)
    assert len(ds13) == 2, f'ожидалось 2 окна, получено {len(ds13)}'
    _, y13 = ds13[1]
    assert y13.shape[0] == 4
    # Окно start=1: targets — строки 6, 8, 10, 12
    expected1 = ds.scalers['targets'].transform(
        (np.array([6, 8, 10, 12]) * 3.0).reshape(-1, 1))
    assert np.allclose(np.asarray(y13), expected1)

    # Шаг 1 — поведение как раньше (backward compat)
    cfg1 = Cfg()
    cfg1.sequence_length = 5
    cfg1.prediction_horizon = 4
    cfg1.prediction_step = 1
    cfg1.feature_columns = ['a']
    cfg1.target_columns = ['b']
    ds1 = VesselDataset(seg13, cfg1, fit_scalers=False, scalers=ds.scalers)
    x1, y1 = ds1[0]
    assert y1.shape[0] == 4
    expected_s1 = ds.scalers['targets'].transform(
        (np.arange(5.0, 9.0) * 3.0).reshape(-1, 1))
    assert np.allclose(np.asarray(y1), expected_s1)

    print('OK  test_dataset_prediction_step')


# ---------------------------------------------------------------------------
# 9. Regression: BUG-LSTM-04 — короткий хвост чанкования не в train
# ---------------------------------------------------------------------------

def test_split_segments_tail():
    try:
        import run_training  # noqa: F401
    except ImportError:
        print('SKIP test_split_segments_tail (нет torch)')
        return

    from run_training import split_segments

    rng = np.random.default_rng(0)
    # 12499 строк: 12 полных чанков + хвост 499 строк (числа из аудита)
    df = pd.DataFrame({f'c{i}': rng.uniform(0, 10, 12499) for i in range(3)})
    train_segs, val_segs, test_segs, test_row_ranges = split_segments(df)

    n_train = sum(len(s) for s in train_segs)
    n_val = sum(len(s) for s in val_segs)
    n_test = sum(len(s) for s in test_segs)
    assert n_train + n_val + n_test == 12 * 1000, \
        f'хвост 499 строк попал в train/val/test: {n_train + n_val + n_test}'

    # Доли внутри каждого чанка: 0.7 / 0.15 / 0.15
    assert abs(n_train / (12 * 1000) - 0.7) < 0.01
    assert abs(n_test / (12 * 1000) - 0.15) < 0.01

    # test_row_ranges покрывают только test-строки обработанных чанков
    for (s, e) in test_row_ranges:
        assert e - s == 150

    # Полный чанк — как раньше
    df2 = pd.DataFrame({'a': rng.uniform(0, 10, 1000)})
    t2, v2, te2, r2 = split_segments(df2)
    assert sum(len(s) for s in t2) == 700
    assert sum(len(s) for s in v2) == 150
    assert sum(len(s) for s in te2) == 150

    # Одиночный маленький df (без полных чанков) — пропорциональный сплит
    # сохраняется (иначе ломались бы quick-проверки пайплайна)
    df_small = pd.DataFrame({'a': rng.uniform(0, 10, 600)})
    t3, v3, te3, r3 = split_segments(df_small)
    assert sum(len(s) for s in t3) == 420   # int(600*0.7)
    assert sum(len(s) for s in te3) == 90   # int(600*0.85) - 420

    # Хвост короче min_chunk (50 < 100 при полных чанках) — отброшен
    df_tail = pd.DataFrame({f'c{i}': rng.uniform(0, 10, 12050) for i in range(2)})
    t4, v4, te4, r4 = split_segments(df_tail)
    assert sum(len(s) for s in t4) + sum(len(s) for s in v4) + \
        sum(len(s) for s in te4) == 12 * 1000

    print('OK  test_split_segments_tail')


# ---------------------------------------------------------------------------
# 10. Regression: BUG-LSTM-07 — round-trip конфигурации из manifest
# ---------------------------------------------------------------------------

def test_config_from_manifest_roundtrip():
    from registry import config_from_manifest

    manifest = {
        'features': {
            'profile': 'motion_core_prediction',
            'input': ['a', 'b'], 'targets': ['c'],
            'target_weights': {'c': 1.5},
        },
        'model': {
            'sequence_length': 60, 'prediction_horizon': 7, 'prediction_step': 2,
            'encoder_hidden_dims': [32], 'temporal_hidden_size': 16,
            'temporal_num_layers': 2, 'decoder_hidden_dim': 16,
            'bidirectional': False, 'use_attention': True,
            'encoder_dropout': 0.2, 'temporal_dropout': 0.3, 'decoder_dropout': 0.05,
        },
        'training': {
            'batch_size': 32, 'learning_rate': 0.0007, 'weight_decay': 3e-4,
            'max_grad_norm': 0.5, 'loss_weights': {'mse': 2.0, 'huber': 0.0,
                                                   'smoothness': 0.05},
            'teacher_forcing_ratio': 0.7,
        },
    }

    cfg = config_from_manifest(manifest)
    assert cfg.profile == 'motion_core_prediction'
    assert cfg.sequence_length == 60
    assert cfg.prediction_horizon == 7
    assert cfg.prediction_step == 2
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.0007
    assert cfg.weight_decay == 3e-4
    assert cfg.max_grad_norm == 0.5
    assert cfg.loss_weights == {'mse': 2.0, 'huber': 0.0, 'smoothness': 0.05}
    assert cfg.teacher_forcing_ratio == 0.7
    assert cfg.temporal_dropout == 0.3
    assert cfg.device == 'cpu'

    # Старый manifest без этих ключей — дефолты, без исключений
    cfg_old = config_from_manifest({
        'features': {'input': ['a'], 'targets': ['b']},
        'model': {'sequence_length': 10, 'prediction_horizon': 3},
    })
    assert cfg_old.sequence_length == 10
    assert cfg_old.prediction_horizon == 3
    assert cfg_old.prediction_step == 1
    assert cfg_old.batch_size == 48

    print('OK  test_config_from_manifest_roundtrip')


if __name__ == '__main__':
    test_feature_engineering_consistency()
    test_dataset_segment_windows()
    test_physical_metrics()
    test_registry()
    test_config_from_manifest()
    test_config_full_profile_engineering()
    test_extract_fe_params_relative_wind()
    test_dataset_prediction_step()
    test_split_segments_tail()
    test_config_from_manifest_roundtrip()
    print('\nВсе smoke-тесты пройдены ✓')
