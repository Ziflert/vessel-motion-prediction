# 🚢 LSTM-Vessel_Motion_Prediction — Полное описание проекта

> **Документ создан для восстановления контекста после длительного перерыва.**
> Здесь собрано всё: что это за модель, какие параметры актуальны, какие получены результаты,
> в каком формате велась работа, и какие «грабли» ждут при повторном запуске.

> ⚡ **ПРОЕКТ ОБНОВЛЁН (см. §12)**: внедрена система версионирования (manifest.json +
> experiments.csv + registry.py), исправлены импорты, утечка на границах чанков, добавлены
> физические метрики/бейзлайны/инженерия признаков (sin/cos углов, относительный курс волны),
> smoke-тесты и git-репозиторий. Разделы §1–§11 описывают состояние ДО обновления
> (актуальны для исторических моделей v001–v004).

---

## 1. Что это за проект (кратко)

Проект предсказывает **движение судна** (качка, поворот, скорость) по историческим данным
бортовых датчиков и погодных условий с помощью нейросети класса **Seq2Seq (Encoder–Decoder на LSTM
с механизмом внимания)**.

**Формат задачи:**
- **Вход:** временно́е окно истории датчиков `sequence_length` шагов (1 шаг = 1 секунда).
- **Выход:** прогноз `prediction_horizon` шагов вперёд для набора целевых переменных.
- **Тип:** многомерная регрессия временных рядов (multi-horizon, multi-target).

Язык интерфейсов и комментариев — русский, названия колонок данных — русские
(например `Roll(градусы)`, `ROT(°/мин)`).

---

## 2. Хронология: что происходило

| Дата | Событие |
|---|---|
| 2025-12-10 | Получены исходные данные `data/raw/your_data.csv` (12 499 строк, TSV/UTF-16) |
| 2025-12-18 (день) | Разработка пайплайна, первый запуск на **CPU** (внутр. версия v003, R²=0.22, 6.3 мин) |
| 2025-12-18 (вечер) | Переход на **CUDA GPU**. Серия экспериментов с горизонтом прогноза (30 → 15 → 10 шагов). Внутренняя нумерация версий дошла до v007 |
| 2025-12-18 (ночь) | Интерактивные inference-эксперименты с разными длиной истории/горизонтом (результаты в `results/inference_v007_*`) |
| 2025-12-18/19 | Модели перенесены в `models_archive/`, **папки переименованы вручную** в v001–v004 (внутренние имена v003, v004, v007, v004 — см. §8) |
| 2025-12-19 | Добавлен профиль `full_prediction` (8 целей). Обучена v004 (474K параметров), выполнен inference по 8 целям |
| 2025-12-19 | Написана документация (`README/reade_me.txt`, `cheat-sheet.txt`, `speed_compare.txt`); старая копия скриптов сохранена в `!version_1/` |
| 2026-01-11 | Проект упакован в `LSTM-Vessel_Motion_Prediction.rar` (полный снимок, 8 МБ) |

**Итог:** за ~2 дня проведено ≥7 обучений, выбрана лучшая модель (внутр. v007 = папка `models_archive/v003_20251218_221330`,
R² = 0.62 на тесте в стандартизованном пространстве), проверена работа full-профиля.

---

## 3. Данные

**Файл:** `data/raw/your_data.csv` (~10 МБ)

| Свойство | Значение |
|---|---|
| Формат | TSV (tab-separated), кодировка **UTF-16** (в коде есть fallback на default) |
| Строк | 12 499 |
| Колонок | 40 (`time` + 39 признаков); `time` удаляется при загрузке |
| Дискретизация | **1 секунда** между измерениями → датасет ≈ 3.5 часа записи |
| Разделение | «Mixed Weather Split (Chunked)»: чанки по 1000 строк, внутри чанка 70% train / 15% val / 15% test; чанки <100 строк → в train. Итог: 8749 / 1875 / 1875 |

**Группы колонок:**
- **Движение:** Pitch, Roll, Vertical, Velocity.Pitching/Rolling/Vertical/Yawing
- **Море/волны:** Swell(метры), Wave.Highest, Wave.current, Wave.direction, Swell.direction, Wave.speed
- **Ветер/течение:** Wind.direction, Current.direction, Current.speed
- **Управление:** Rudder Order, Rudder State, RPM
- **Навигация:** Long, Lat, STW, SOG, ROT, Course
- **Моменты/силы:** Moment Yawing/Rolling/Pitching, Force Vertical/Summary/Longitudinal/Lateral (+ Wind.-аналоги)

⚠ Нормализация: `StandardScaler` отдельно для признаков и для целей; обучается **только на train**,
сохраняется в `checkpoints/scalers.pkl` (dict: `{'features': scaler, 'targets': scaler}`).

⚠ При **inference** данные должны содержать ровно те же колонки, что были при обучении
(список хранится в чекпоинте в `config.feature_columns`).

---

## 4. Архитектура модели

Модель — **Seq2Seq с вниманием (additive/Bahdanau-подобное)**: `models/vessel_predictor.py`.

```
Вход [batch, seq_len, input_dim]
   │
   ▼
ENCODER (models/encoder.py)
   • Feature extractor: Linear(input_dim → 128) → ReLU → Dropout(0.15) → Linear(128→96) → ReLU → Dropout
   • LSTM(hidden=128, layers=2, dropout=0.1, batch_first, НЕ bidirectional)
   → outputs [batch, seq_len, 128], hidden (h_n, c_n)
   │
   ▼
BRIDGE (vessel_predictor.py)
   • Два Linear-слоя bridge_h / bridge_c: 128 → 96 (проецируют h и c энкодера под декодер)
   │
   ▼
DECODER (models/decoder.py) — авторегрессионный, шаг за шагом, prediction_horizon итераций
   • Вход шага: prev_target [batch, output_dim] ⊕ context vector от ATTENTION [batch, 128]
   • Attention (models/attention.py): concat(hidden_decoder, encoder_outputs) → Linear → tanh → Linear→1 → softmax → взвешенная сумма encoder_outputs
   • LSTM(hidden=96, layers=2)
   • fc_out: Linear(96 → output_dim)
   • Первый вход декодера — нулевой вектор
   │
   ▼
Выход [batch, prediction_horizon, output_dim]
```

**Teacher forcing:** при обучении `ratio = 0.5` (модель с вероятностью 50% получает реальное
предыдущее значение вместо собственного прогноза). Валидация и inference — всегда без teacher forcing
(полностью авторегрессионно).

**Не используемые, но присутствующие компоненты:** `models/temporal_block.py`
(`LSTMBlock`, `TransformerBlock`, `HybridBlock`, `PositionalEncoding`) — заготовки альтернативных
блоков, в `VesselPredictor` не подключены. Параметры `num_attention_heads`,
`transformer_ff_dim`, `transformer_num_layers` из конфига тоже не используются.

**Почему inference может менять `sequence_length` и `prediction_horizon`:** LSTM-энкодер принимает
окно любой длины, attention работает по всем выходам энкодера, а декодер просто крутит цикл
`prediction_horizon` раз. Поэтому одну и ту же модель можно запускать с другой длиной истории/горизонта
(так и делали — см. §9), но качество на непривычных горизонтах падает.

---

## 5. Актуальная конфигурация (`config/config.py` — состояние последнего обучения v004)

| Параметр | Значение | Комментарий |
|---|---|---|
| `profile` | `full_prediction` | motion / rot / speed / full_prediction / custom |
| `sequence_length` | **120** шагов (2 мин истории) | |
| `prediction_horizon` | **20** шагов (20 сек прогноза) | |
| `prediction_step` | 1 | |
| `encoder_hidden_dims` | [128, 96] | MLP перед LSTM |
| `encoder_dropout` | 0.15 | |
| `temporal_hidden_size` | **128** | LSTM энкодера |
| `temporal_num_layers` | **2** | |
| `temporal_dropout` | 0.1 | |
| `bidirectional` | False | (включение удваивает вычисления) |
| `decoder_hidden_dim` | **96** | LSTM декодера |
| `decoder_dropout` | 0.1 | |
| `use_attention` | **True** | даёт качество, замедляет ~20% |
| `batch_size` | 48 | |
| `learning_rate` | 0.0005 | AdamW |
| `weight_decay` | 1e-4 | |
| `num_epochs` | 150 (макс.) | реально рано срабатывает early stopping |
| `early_stopping_patience` | 25 | по val loss |
| `max_grad_norm` | 1.0 | gradient clipping |
| `num_workers` | 0 | обязательно для Windows |
| `seed` | 42 | |
| `device` | `cuda` при наличии | cuDNN benchmark включён |

**Профиль full_prediction (актуальный):**
- **Признаки (19):** Swell, Wave.Highest, Wave.direction, Wave.speed, Wind.direction, Current.direction,
  Current.speed, Rudder Order, Rudder State, RPM, Pitch, Roll, Vertical, Velocity.Pitching,
  Velocity.Rolling, Velocity.Vertical, SOG, ROT, Course
- **Цели (8) и веса в loss:**

| Цель | Вес |
|---|---|
| Roll(градусы) | **2.0** (самое важное) |
| Vertical(Метр) | 1.5 |
| ROT(°/мин) | 1.5 |
| Pitch(градусы) | 1.0 |
| SOG(узлы) | 1.0 |
| Velocity.Rolling(°/мин) | 1.2 |
| Velocity.Pitching(°/мин) | 0.8 |
| Velocity.Vertical(узлы) | 0.8 |

**Другие профили** (`get_profile_config()` в `config/config.py`):
- `motion_prediction` — 6 целей (качка), признаки: погода + управление + состояние качки + SOG
- `rot_prediction` — 1 цель ROT(°/мин); важно: для ROT нужны Rudder Order/State, SOG, Wave/Current direction
- `speed_prediction` — 1 цель SOG(узлы)
- `custom` — ручные списки в config.py

**Профили/режимы скорости** переключаются утилитой `config/switch_config.py` (она **regex-ом правит
config.py** на диске!):

| Режим | seq | horizon | batch | Оценка времени |
|---|---|---|---|---|
| fast | 120 | 20 | 48 | ~1 час |
| balanced | 180 | 30 | 32 | ~2–3 часа |
| quality | 300 | 60 | 32 | ~6–8 часов |

**Функция потерь** (`training/losses.py`, класс `VesselLoss`):

```
total = 1.0 * MSE(взвешенный по целям)
      + 0.5 * Huber/SmoothL1(взвешенный)
      + 0.1 * temporal smoothness  = mean( (Δpred − Δtarget)² )
```

**Оптимизационная обвязка** (`training/trainer.py`): AdamW(lr 5e-4, wd 1e-4) +
`ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` + grad clip 1.0 + early stopping (25 эпох).
Checkpoint сохраняется при улучшении val loss. Логируется «Top 3 worst variables» (худшие цели по MSE).

⚠ **Метрики обучения (MAE/RMSE/R²) считаются в СТАНДАРТИЗОВАННОМ (z-score) пространстве**,
т.к. сравниваются масштабированные тензоры. Это НЕ физические единицы! Физические метрики
получаются только через `run_inference.py` (там применяется `inverse_transform`).

---

## 6. Формат работы (workflow)

### 6.1 Типовой цикл

```bash
# 0. Посмотреть текущую конфигурацию
python config/switch_config.py            # current config
python config/switch_config.py list       # все профили и режимы

# 1. Выбрать профиль и режим
python config/switch_config.py profile motion   # motion | rot | speed | full
python config/switch_config.py speed fast       # fast | balanced | quality

# 2. Обучить (создаёт новую версию в models_archive/)
python run_training.py

# 3. Inference / визуальная проверка (интерактивный диалог)
python run_inference.py --model-version 3   # или --model-path <путь>, или интерактивно

# 4. Управление версиями
python model_manager.py list
python model_manager.py best --metric r2
python model_manager.py info --version v003_20251218_221330
python model_manager.py compare v001_... v003_...
python model_manager.py export --version vXXX --output <dir>
python model_manager.py delete --version vXXX
```

### 6.2 Что делает `run_training.py`

1. Создаёт папку `models_archive/v{NNN}_{YYYYMMDD_HHMMSS}/` с подпапками `checkpoints/`, `logs/`
   (номер версии = max существующий + 1).
2. Загружает `data/raw/your_data.csv` (TSV/UTF-16), удаляет колонку `time`.
3. Chunked split 70/15/15 (см. §3).
4. Строит Dataset/DataLoader (StandardScaler по train), печатает размерности и число батчей.
5. Создаёт `VesselPredictor`, сохраняет `training_config.json`.
6. Обучает (Trainer), сохраняет `checkpoints/best_model.pt` (лучшая по val loss).
7. Загружает лучшую модель, считает **test-метрики** (в стандартизованном пространстве).
8. Сохраняет: `checkpoints/scalers.pkl`, `training_summary.txt`, `training_metrics.json`, `README.txt`.

### 6.3 Что делает `run_inference.py`

- **Интерактивный скрипт.** Спрашивает: длину истории в шагах (по умолчанию 600),
  горизонт прогноза (по умолчанию 100), число тестов (по умолчанию 5), режим выбора точек
  (even / manual / random).
- Загружает модель + scalers из `models_archive/`, **позволяет переопределить** seq_len и горизонт
  (см. §4 — это работает благодаря seq2seq).
- Для каждой тестовой точки: строит график (история серым, прогноз синим, факт красным,
  MAE/RMSE/MAPE/Corr на графике), сохраняет `prediction_{idx}.png`.
- Пишет в папку `results/inference_{модель}_{timestamp}/`: `parameters.txt`,
  `predictions.csv` (pred_/actual_/error_ колонки по шагам), `summary_statistics.txt`.

### 6.4 Версионирование моделей

Каждое обучение → отдельная папка в `models_archive/` с полным набором артефактов:
модель, scalers, конфиг, метрики, сводка. **Инференс всегда берёт конфиг из чекпоинта**
(`checkpoint['config']` — pickled-объект `Config`), а не из текущего `config.py`, поэтому признаки/цели
всегда правильные для конкретной модели.

⚠ Внутри checkpoint'а лежит pickled `Config` → загрузка требует того же класса
(`config.config.Config`) и совместимого Python (см. §11).

---

## 7. Результаты

### 7.1 Обучение (test-метрики, СТАНДАРТИЗОВАННОЕ пространство, усреднение по всем целям и шагам)

| Папка в models_archive | Внутр. имя | Профиль/цели | Seq/Pred | Параметры | Эпох | Устройство | Время | Val Loss | Test Loss | MAE | RMSE | R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v001_20251218_171905 | v003 | motion-подобный, 5 целей | 120/30 | 85 957 | 27 | CPU | 6.3 мин | 1.0531 | 0.9175 | 0.5469 | 0.8724 | **0.2202** |
| v002_20251218_220910 | v004 | 5 целей | 120/**15** | 85 957 | 28 | CUDA | 1.7 мин | 0.7546 | 0.6671 | 0.4411 | 0.7220 | 0.4672 |
| **v003_20251218_221330** | **v007** | 5 целей | 120/**10** | 85 957 | 36 | CUDA | 1.6 мин | **0.5508** | **0.4903** | **0.3725** | **0.6096** | **0.6208** ⭐ |
| v004_20251219_120419 | v004 | **full, 8 целей** | 120/20 | 474 344 | 40 | CUDA | 5.9 мин | 0.9391 | 0.8649 | 0.4636 | 0.7542 | 0.4382 |

**Ключевой вывод экспериментов:** уменьшение горизонта прогноза (30 → 15 → 10 шагов) при той же
маленькой архитектуре (64-скрытые, 1 слой, lr 0.001, batch 64 — конфигурация той серии) дало
главный прирост качества: **R² 0.22 → 0.47 → 0.62**. Лучшая модель — короткий горизонт 10 сек.

Архитектура ранних версий (v001–v003): encoder MLP [64], LSTM 64×1, decoder 64, lr 0.001,
batch 64, epochs 100, patience 20. Целей — 5 (Pitch, Roll, Vertical, Vel.Pitching, Vel.Rolling —
без Velocity.Vertical), признаков 17.

### 7.2 Inference (физические единицы, после inverse_transform)

Модель **v003 (внутр. v007), 5 целей** — чувствительность к настройке inference:

| Запуск | История/Горизонт | Ср. MAE | Ср. RMSE | Комментарий |
|---|---|---|---|---|
| results/inference_v007_..._222525 | 600 шагов / 100 сек | 9.36 | 21.61 | очень плохо (чужой горизонт) |
| results/inference_v007_..._223958 | 60 / 10 | 7.87 | 18.23 | близко к обучению |
| results/inference_v003_..._230132 | 30 / 15 | **6.66** | 16.07 | лучший запуск; Best MAE 0.44 (idx 2105) |

Per-target MAE в лучшем запуске (v003, seq30/pred15):
Pitch **0.072°**, Roll **0.83°**, Vertical **0.40 м**, Vel.Pitching **2.65°/мин**, Vel.Rolling **29.4°/мин**.

Модель **v004 (full_prediction, 8 целей)** — запуск seq60/pred15, ср. MAE 4.97:

| Цель | MAE (физ. ед.) |
|---|---|
| Pitch(градусы) | 0.085° |
| Roll(градусы) | 1.02° |
| Vertical(Метр) | 0.38 м |
| Velocity.Pitching(°/мин) | 3.28 |
| Velocity.Rolling(°/мин) | **33.0** ← доминирует в средней ошибке |
| Velocity.Vertical(узлы) | 0.60 |
| ROT(°/мин) | 1.24 |
| SOG(узлы) | 0.087 |

**Выводы:**
- Позиционные величины (Pitch, Roll, Vertical, SOG) предсказываются хорошо.
- **Velocity.Rolling — стабильно худшая переменная** (огромные ошибки 29–37 °/мин во всех запусках),
  она же завышает средний MAE. Кандидат на исключение из целей или на понижение веса / отдельную модель.
- Прогноз сильно деградирует на горизонтах, сильно отличающихся от обучающего.
- Качка предсказывается лучше ROT (ROT зависит от решений экипажа — непредсказуемая составляющая).

---

## 8. Карта репозитория

```
LSTM-Vessel_Motion_Prediction/
├── config/
│   ├── config.py            # ВСЕ параметры + определения профилей (правится switch_config.py)
│   └── switch_config.py     # CLI переключения profile/speed (regex-правка config.py)
├── data/
│   ├── dataset.py           # VesselDataset + create_dataloaders (StandardScaler, float32)
│   └── raw/your_data.csv    # ИСХОДНЫЕ ДАННЫЕ (TSV, UTF-16, 12499×40, 1 Гц)
├── models/
│   ├── vessel_predictor.py  # Seq2Seq сборка (encoder→bridge→decoder, teacher forcing)
│   ├── encoder.py           # MLP + LSTM
│   ├── decoder.py           # LSTM-декодер с attention
│   ├── attention.py         # Attention (concat → tanh → softmax)
│   └── temporal_block.py    # НЕ ИСПОЛЬЗУЕТСЯ (заготовки LSTM/Transformer/Hybrid блоков)
├── training/
│   ├── trainer.py           # Trainer: AdamW, ReduceLROnPlateau, early stop, чекпоинты
│   ├── losses.py            # VesselLoss: MSE + 0.5*Huber + 0.1*smoothness, веса целей
│   └── metrics.py           # MAE / RMSE / R² (по flatten)
├── scripts/
│   ├── check_data.py        # проверка формата CSV (кодировки/разделители)
│   └── diagnose_perfomance.py  # диагностика CUDA, benchmark CPU vs GPU
├── run_training.py          # обучение + версионирование в models_archive/
├── run_inference.py         # интерактивный inference + графики + CSV
├── model_manager.py         # CLI менеджер версий (list/info/best/compare/export/delete)
├── models_archive/          # v001–v004: checkpoints/best_model.pt, scalers.pkl, конфиги, метрики
├── results/                 # 4 папки inference: png-графики, predictions.csv, statistics
├── README/                  # reade_me.txt (гайд), cheat-sheet.txt (шпаргалка),
│                            # speed_compare.txt (анализ скорости), requirements.txt
├── !version_1/              # БЭКАП старых версий run_training/run_inference/model_manager
├── .idea/                   # PyCharm (Python 3.11, venv "LSTM-Vessel_Motion_Prediction")
├── .venv/                   # СЛОМАН (см. §11): базовый Python 3.11 удалён
├── __init__.py              # пустой (заглушка пакета)
└── LSTM-Vessel_Motion_Prediction.rar  # архивный снимок проекта (2026-01-11)
```

### Несоответствие имён версий (важно!)

Папки `models_archive/` переименовывались вручную после обучения. Внутри файлов
(`training_summary.txt`, `README.txt`, `training_metrics.json`) остались **исходные внутренние имена**:

| Папка (используется `--model-version N`) | Внутреннее имя в файлах |
|---|---|
| v001_20251218_171905 | v003 |
| v002_20251218_220910 | v004 |
| v003_20251218_221330 | **v007** |
| v004_20251219_120419 | v004 |

Поэтому `results/inference_v007_20251218_221330_*` — это inference той же модели, что сейчас лежит
в папке `models_archive/v003_20251218_221330`. Номер версии в чекпоинте (`checkpoint['config']`)
— «внутренний», не совпадает с именем папки.

---

## 9. Окружение и запуск заново

### 9.1 Окружение на момент последней успешной работы

| Компонент | Версия |
|---|---|
| Python | **3.11.6** (CPython, Windows) — ВАЖНО для чтения pickled чекпоинтов |
| virtualenv | 20.31.2 |
| torch | **2.7.1+cu118** (CUDA-сборка), torchvision 0.22.1+cu118 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.1 |

Полный (избыточно широкий) список зависимостей: `README/requirements.txt`.
Реально нужны: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm` (+ `tabulate` для
`model_manager.py list`).

### 9.2 Восстановление окружения

`.venv` в проекте **сломан** — он ссылается на базовый интерпретатор
`C:\Users\Dmitrii Zifert\AppData\Local\Programs\Python\Python311\python.exe`, которого больше нет.
Пересоздать:

```powershell
py -3.11 -m venv .venv            # или любой Python 3.11 (для чтения чекпоинтов — желательно 3.11)
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib tqdm tabulate
```

Без GPU можно поставить CPU-сборку torch (обучение будет ~в 5–10 раз медленнее,
но маленькая модель 86K параметров обучалась и на CPU за минуты).

### 9.3 Проверка

```bash
python scripts/check_data.py              # формат данных
python scripts/diagnose_perfomance.py     # CUDA/скорость
python config/switch_config.py            # текущая конфигурация
python model_manager.py list              # 4 сохранённые модели должны быть видны
python run_inference.py --model-version 3 # проверка загрузки лучшей модели
```

---

## 10. Известные проблемы и «грабли» (обязательно прочитать)

1. **Импорты `version_1` сломаны в текущей раскладке.**
   `models/vessel_predictor.py`, `models/decoder.py` и `scripts/diagnose_perfomance.py` импортируют
   `from version_1.models...` — проект когда-то назывался/лежал как пакет `version_1`.
   Сейчас папка называется `LSTM-Vessel_Motion_Prediction`, поэтому при запуске будет
   `ModuleNotFoundError: No module named 'version_1'`.
   **Фикс (выбрать одно):**
   - заменить импорты на относительные: `from .models.encoder import Encoder`,
     `from .models.attention import Attention` и т.д.;
   - либо положить проект в родительский каталог как папку `version_1`;
   - либо добавить в начало скриптов `sys.path`-костыль/переименовать пакет.
2. **`.venv` сломан** (базовый Python 3.11 удалён) — см. §9.2.
3. **Метрики обучения — в стандартизованном пространстве.** MAE 0.37 ≠ 0.37 физической величины.
   Не сравнивать напрямую с метриками inference!
4. **Ручное переименование папок моделей** (§8) — внутренние версии не совпадают с именами папок.
   При новом обучении `get_next_version_number()` продолжит нумерацию с v005 (по именам папок v004 → next = 5).
5. **Chunked split может подтекать:** окна последовательностей нарезаются по всему DataFrame без учёта
   границ чанков train/val/test — соседние окна на стыках частично перекрываются (небольшая утечка
   между выборками). Для честной оценки можно резать датасет по границам чанков.
6. **`weights_only=False`** в `torch.load` (trainer и inference): чекпоинты содержат pickled `Config`
   — грузить только из доверенных файлов, и Python должен быть совместим (проверено на 3.11).
7. **`num_workers: 0`** — на Windows иначе не работает; не поднимать без нужды.
8. **Velocity.Rolling портит средние метрики** — исключить из целей или понизить вес, если она не нужна.
9. **`switch_config.py` молча правит `config/config.py` regex-ом** — при нестандартном форматировании
   файла может не найти/неправильно заменить значения. После переключения проверять файл глазами.
10. **Веса целей (`target_weights`) не сохраняются в `training_config.json`** — полный словарь весов
    есть только внутри pickled-конфига в чекпоинте и в текущем `config.py`.
11. **Русские названия колонок** — при работе с данными на других машинах следить за кодировкой
    (UTF-16) и точным совпадением имён колонок (включая скобки и единицы измерения).
12. Старые запуски inference создавались **другой версией `run_inference.py`** (без секции
    «Model profile» и «Original model training parameters» в parameters.txt) — текущая версия скрипта
    более информативная. Разница с бэкапом — в `!version_1/run_inference.py`.

---

## 11. Быстрая шпаргалка «что где какое качество»

- **Лучшая модель:** `models_archive/v003_20251218_221330` (внутр. v007).
  Маленькая (86K параметров), горизонт 10 сек, test R² = 0.62 (стандартиз.),
  физически: Pitch ~0.07°, Roll ~0.8–1.0°, Vertical ~0.3–0.4 м на горизонте 10–15 сек.
- **Самая универсальная:** `models_archive/v004_20251219_120419` — full_prediction, 8 целей
  (качка + ROT + SOG), 474K параметров, test R² = 0.44.
- **Типовой inference:** история 30–60 шагов, горизонт ≤ 15 шагов — там лучшие метрики.
- **Не использовать:** горизонт 100 шагов с моделями, обученными на 10–20 (качество разваливается).

---

## 12. Обновление проекта (текущее состояние): версионирование + P0-правки

### 12.1 Новая система версионирования (registry)

**Двухуровневая схема** (реализована в `registry.py`):

1. **`manifest.json`** — карточка каждого запуска внутри папки версии. Пишется ДО обучения,
   результаты дописываются после. Содержит: run_id, статус, гипотезу, git-коммит, sha256 данных,
   полный снапшот условий (модель/обучение/признаки/веса), параметры скалеров целей (mean/std —
   БЕЗ pickle), результаты (масштабированные + физические + бейзлайны + skill score).
2. **`models_archive/experiments.csv`** — реестр-оглавление: одна строка на запуск
   (run_id, профиль, цели, seq/horizon, метрики, skill, статус, заметки, путь).

**Правила:** run_id = `YYYYMMDD-HHMMSS-xxxx`, папки НИКОГДА не переименовываются;
статусы `candidate`/`production`/`archived` (production — ровно одна модель);
inference грузит конфигурацию из manifest.json (`config_from_manifest`), а не из pickle чекпоинта.

**Legacy-модели v001–v004 мигрированы** (`scripts/migrate_models_archive.py`): для каждой создан
manifest.json (run_id = имя папки, status = archived), веса целей реконструированы из профилей
(помечены `target_weights_source: reconstructed...`).

**Команды:**

```bash
python run_training.py --notes "проверяю horizon=10"     # обучение: создаёт run + manifest + строку в реестре
python run_inference.py --run-id production              # или --run-id 20251219-..., префикс, или 'latest'
python run_inference.py --model-version 3                # legacy-совместимость (папка v003_*)
python model_manager.py list                             # таблица всех моделей (+статусы)
python scripts/migrate_models_archive.py                 # миграция legacy-папок (идемпотентна)
py -3.13 tests/test_smoke.py                             # smoke-тесты (без torch)
```

### 12.2 Исправленные проблемы

| Проблема | Решение |
|---|---|
| Импорты `from version_1.models...` (ModuleNotFoundError) | Заменены на относительные (`from .models...`), в scripts — на прямые |
| Утечка/склейка окон на границах чанков сплита | `split_segments` возвращает НЕПРЕРЫВНЫЕ сегменты; `VesselDataset` нарезает окна строго внутри сегмента (проверено тестом) |
| Метрики только в z-score пространстве | `training/metrics.py: full_physical_report` — MAE/RMSE/R² в физических единицах, per-target и per-horizon (кривая ошибки по времени упреждения) |
| Нет базы сравнения | `training/baselines.py`: persistence + линейная экстраполяция; skill score = 1 − MAE_модели/MAE_бейзлайна |
| Разрыв 0°/360° в углах | `data/features.py`: все направленческие углы → sin/cos; добавлен **относительный угол встречи волны** (Wave.direction − Course) как sin/cos. Флаги `cyclic_encoding`, `relative_wave_angle` в Config; применяется одинаково при обучении и inference, фиксируется в manifest |
| Velocity.Rolling портит метрики | Новый профиль `motion_core_prediction` (только Pitch/Roll/Vertical, без зашумлённых производных): `python config/switch_config.py profile core` |
| Attention-веса выбрасывались | `Attention.forward` возвращает weights; `VesselPredictor.last_attention_weights` — список весов по шагам прогноза (для интерпретации) |
| Падение печати R²/эмодзи в cp1251-консоли | `_force_utf8_stdio()` во всех CLI-точках входа |
| Воспроизводимость | seed (torch/numpy/random) в run_training; sha256 данных, git-коммит, pip freeze в manifest/environment.txt |

### 12.3 Формат manifest.json (кратко)

```json
{
  "run_id": "20251219-120419-a3f2",
  "status": "candidate",
  "hypothesis": "что проверяем",
  "git_commit": "...", "data": {"sha256": "...", "rows_train": "...", "split": {...}},
  "model": {"sequence_length": 120, "prediction_horizon": 20, ...},
  "training": {"learning_rate": 0.0005, "seed": 42, ...},
  "features": {"input": [...], "targets": [...], "target_weights": {...},
               "feature_engineering": {"cyclic_encoding": true, "relative_wave_angle": true}},
  "scalers": {"target_mean": [...], "target_std": [...]},
  "results": {"scaled_test": {...}, "physical": {"overall": {...}, "per_target": {...},
               "per_horizon_mae": [...]}, "baselines": {...}, "skill_vs_persistence": {...}}
}
```

### 12.4 Git

Репозиторий: `git@github.com:Ziflert/vessel-motion-prediction.git`.
В git НЕ входят: `.venv/`, `__pycache__/`, `.idea/`, `*.rar` (см. `.gitignore`).
Данные и обученные модели входят (репозиторий самодостаточен).

### 12.5 Инструменты исследования качки (P1, реализованы)

| Скрипт | Что делает |
|---|---|
| `scripts/evaluate_regimes.py` | **Разбор по режимам волнения**: MAE/RMSE по бакетам высоты волны (calm/moderate/rough/severe) и угла встречи волны (following/quartering/beam/head). Для новых моделей автоматически берёт test-строки из manifest, для legacy — вся выборка (с пометкой in-sample) |
| `scripts/rolling_eval.py` | **Rolling-прогноз + спектральный анализ**: непрерывный прогноз с шагом 1 с, persistence-бейзлайн, периодограммы (Уэлч) факта/прогноза, фазовый сдвиг через взаимную корреляцию, амплитудный коэффициент. График: временные ряды + PSD |
| `scripts/compare_models.py` | **Сравнение моделей на одинаковых окнах** (одна сетка окон, единый горизонт) — корректная альтернатива разнобою старых inference-запусков |
| `run_inference.py --mc-samples N` | **MC-Dropout неопределённость**: средний прогноз ± std в predictions.csv |
| `run_training.py --max-epochs/--data-limit/--seed` | Быстрые проверки пайплайна и мультисид-эксперименты |
| `registry.delete_run/rebuild_registry` | Управление реестром (CSV всегда пересобираем из manifests) |

### 12.6 Первые содержательные результаты новых инструментов (run v003/v007, in-sample)

- **Режимы волнения**: severe (>2.5 м) MAE ≈ 7.2 против rough (1.5–2.5 м) ≈ 0.82 — **в 9 раз хуже**;
  наибольшая ошибка на beam seas (лаговая волна 60–120°). Большая часть записи — тяжёлые погодные
  условия (severe) — это основной режим данных.
- **Rolling-прогноз Roll, упреждение 5 с**: MAE 1.20° (persistence 6.38°), **skill +0.81**,
  корреляция 0.948, **фазовый сдвиг 0.0 с** — модель не «плывёт» по фазе;
  пик спектра: факт T = 11.1 с, прогноз T = 12.2 с — собственный период бортовой качки захвачен;
  амплитудный коэффициент 0.90 (лёгкое занижение размаха).
- **Сравнение на одинаковых окнах** (rows 10000–12499, horizon 10): v004 (full, 474K) MAE 5.87
  против v003 10.59 — НО по Roll v004 лучше (1.00° против 1.21°), по Pitch сопоставимы;
  доминирует по-прежнему Velocity.Rolling.
- **MC-Dropout**: наибольшая неопределённость — у Velocity.Rolling (согласуется с качеством).
- v001 несовместима с текущим кодом (чекпоинт архитектуры до bridge-слоёв) — помечена в manifest.

### 12.7 Окружение (восстановлено)

`.venv` пересоздан на **Python 3.13.15** + **torch 2.7.1+cu118** (CUDA RTX 4070 — GPU-обучение работает).
Зависимости: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib, tqdm, tabulate.
Legacy-скалеры (sklearn 1.8.0) загружаются с InconsistentVersionWarning — безопасно,
но при переобучении версии зафиксируются в `environment.txt` каждого запуска.

### 12.8 Что осталось из бэклога

- Мультисид-прогоны (3–5 сидов, mean±std) — механика готова (`--seed`), нужно только время GPU;
- абляции seq_len/horizon/attention через реестр;
- визуализация attention-весов (веса уже возвращаются моделью);
- честная оценка legacy-моделей вне train-данных (для v001–v003 границы test неизвестны).

---

*Документ сгенерирован по состоянию проекта после ревизии кода, конфигов,
`models_archive/` (training_config.json, training_summary.txt) и `results/` (parameters.txt,
summary_statistics.txt). Все числа взяты из файлов проекта. §12 — по состоянию после рефакторинга.*
