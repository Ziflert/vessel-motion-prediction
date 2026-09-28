"""
Unit-тесты ШАГА 2 — детекторы аномалий (online/anomaly.py).

Запуск (без torch, как test_smoke.py):
    py -3.13 tests/test_anomaly.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config.online import OnlineConfig
from online.anomaly import (Hysteresis, InputAnomalyDetector, ErrorMonitor,
                            baseline_mae_from_manifest)


class FakeScaler:
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale


def test_hysteresis():
    h = Hysteresis(consecutive=2, cooldown_ticks=10)
    assert h.update(True, 0) is False, 'одиночное срабатывание не должно включать маркер'
    assert h.update(True, 1) is True, 'два подряд — включение'

    # cooldown: быстрое выключение запрещено
    assert h.update(False, 2) is True
    assert h.update(False, 3) is True, 'cooldown: статус не может смениться раньше 10 tick'
    assert h.update(False, 12) is False, 'после cooldown — выключение'

    # одиночный сброс не должен гасить устойчивый маркер
    h2 = Hysteresis(consecutive=2, cooldown_ticks=0)
    h2.update(True, 0); h2.update(True, 1); h2.update(True, 2)
    h2.update(False, 3)
    assert h2.update(True, 4) is True, 'одиночный сброс не должен гасить маркер'
    print('  ✓ Hysteresis')


def test_input_anomaly_detector():
    cfg = OnlineConfig()
    # 2 признака: нормальные значения 0 и 10, std 1 и 2
    det = InputAnomalyDetector(['a', 'b'], FakeScaler([0.0, 10.0], [1.0, 2.0]), cfg)

    ok = det.check([0.5, 11.0], tick=0)
    assert not ok['triggered'], 'типичная строка не должна срабатывать'

    bad = det.check([0.5, 10.0 + 10 * 2.0], tick=0)  # b: z=10
    assert bad['triggered'] and bad['features'] == ['b']
    assert abs(bad['max_z'] - 10.0) < 1e-6
    print('  ✓ InputAnomalyDetector: выброс ловится, имя признака корректно')

    # вырожденный признак (std=0) не должен ломать детектор
    det0 = InputAnomalyDetector(['a', 'const'], FakeScaler([0.0, 5.0], [1.0, 0.0]), cfg)
    res = det0.check([0.0, 5.0], tick=0)
    assert not res['triggered'], 'std=0: константа не срабатывает'
    print('  ✓ InputAnomalyDetector: вырожденный признак безопасен')

    # hysteresis внутри detector'а: маркер включается после 2 подряд
    # (bad выше был 1-м срабатыванием — эта строка 2-е подряд)
    res = det.check([0.5, 30.0], tick=0)
    assert res['triggered'] and res['marker'], 'второе подряд срабатывание — маркер включён'
    res = det.check([0.5, 30.0], tick=1)
    assert res['marker'], 'устойчивое срабатывание держит маркер'
    print('  ✓ InputAnomalyDetector: hysteresis маркера')


def test_error_monitor():
    cfg = OnlineConfig(error_anomaly_k=2.5, hysteresis_consecutive=2,
                       marker_cooldown_s=0.0)
    mon = ErrorMonitor([1.0, 2.0], cfg)

    # типичная ошибка: ratio ~0.5 — маркер не горит
    for t in range(4):
        mon.observe(t, [0.5, 1.0])
    assert not mon.marker
    print('  ✓ ErrorMonitor: типичная ошибка — маркер выключен')

    # устойчивое превышение x5 по одной цели — маркер должен включиться
    for t in range(10, 13):
        r = mon.observe(t, [0.5, 10.0])
    assert mon.marker, 'устойчивое превышение порога должно включить маркер'
    assert r['ratio'] > cfg.error_anomaly_k
    print('  ✓ ErrorMonitor: устойчивое превышение включает маркер')


def test_baseline_from_manifest():
    m = {'results': {'physical': {'per_target': {'Roll(градусы)': {'mae': 1.5}}}}}
    assert baseline_mae_from_manifest(m, ['Roll(градусы)']) == [1.5]
    assert baseline_mae_from_manifest(None, ['x']) is None
    assert baseline_mae_from_manifest({}, ['x']) is None
    assert baseline_mae_from_manifest({'results': {}}, ['x']) is None
    print('  ✓ baseline_mae_from_manifest: nominal + все fallback')


if __name__ == '__main__':
    test_hysteresis()
    test_input_anomaly_detector()
    test_error_monitor()
    test_baseline_from_manifest()
    print('\nOK — все тесты ШАГА 2 пройдены')
