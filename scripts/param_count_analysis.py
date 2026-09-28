"""
Расчёт числа параметров LSTM-модели в зависимости от набора входных признаков
(вопрос заказчика 2026-09-28): расширение признаков speed-course-КУ wind-КУ wave-RPM
и минимально необходимых (высота волны, скорость ветра, T_w и др.).

Фактический текущий вход (full_prediction после инженерии): 25 переменных
(19 сырых каналов; 4 угла → sin/cos; + относительный угол волны sin/cos),
параметров 475,112 (подтверждено обучением 20260928-073913).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from config.config import Config
from models.vessel_predictor import VesselPredictor

TARGETS = [
    'Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)',
    'Velocity.Pitching(°/мин)', 'Velocity.Rolling(°/мин)', 'Velocity.Vertical(узлы)',
    'ROT(°/мин)', 'SOG(узлы)',
]


def count_params(n_input: int, label: str):
    config = Config()
    config.verbose = False
    config.target_columns = list(TARGETS)
    config.feature_columns = [f'f{i}' for i in range(n_input)]
    model = VesselPredictor(input_dim=n_input, output_dim=len(TARGETS), config=config)
    total = sum(p.numel() for p in model.parameters())
    print(f'{label:50s} вход={n_input:2d}  параметров={total:>9,}')
    return total


print('=' * 78)
print('Число параметров модели vs размер входа (цели фиксированы: 8)')
print('=' * 78)
t25 = count_params(25, 'текущий вход (19 сырых каналов после инженерии)')
t30 = count_params(30, '+ Wind.speed, T_w, Wave.current (5 каналов)')
t35 = count_params(35, '+ ветер/течение детально, моменты (ещё 5)')
t45 = count_params(45, 'полный формат CSV (~39 сырых → ~45 после инженерии)')

print()
print(f'Прирост 25→30 входов:  +{t30 - t25:,} параметров ({(t30 - t25) / t25 * 100:.2f} %)')
print(f'Прирост 25→35 входов:  +{t35 - t25:,} параметров ({(t35 - t25) / t25 * 100:.2f} %)')
print(f'Прирост 25→45 входов:  +{t45 - t25:,} параметров ({(t45 - t25) / t25 * 100:.2f} %)')
print()
print('Почему так мало: параметры LSTM-слоя = 4·(H·D + H² + 2H), где H — скрытый размер')
print('(128/96, фиксирован), D — число входов: зависимость от D ЛИНЕЙНАЯ.')
print('Расширение набора признаков почти не меняет размер модели.')
