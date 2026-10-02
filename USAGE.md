# 📖 USAGE — Как пользоваться проектом (памятка)

> **Сначала прочтите `CONCEPT.md`** — каноническое описание замысла (что строим и зачем).
> Краткий практический гайд: что запускать, где какие параметры, где смотреть результаты.
> Полный контекст проекта — в `PROJECT_OVERVIEW.md`, журнал исследований — `RESEARCH_LOG.md`.

---

## 0. Окружение

`.venv` уже создан (Python 3.13 + torch 2.7.1+cu118, работает GPU RTX 4070).

```powershell
cd D:\Temprary\LSTM-Vessel_Motion_Prediction
.venv\Scripts\activate          # активировать окружение

# все команды ниже — с активным окружением (или вместо python писать .venv\Scripts\python.exe)
python config/switch_config.py  # проверка: показать текущую конфигурацию
```

---

## 1. ГЛАВНЫЙ ФАЙЛ ПАРАМЕТРОВ — `config/config.py`

Всё настраивается здесь. Что менять под задачу:

| Параметр | Что делает | Когда менять |
|---|---|---|
| `profile` | Какие цели предсказываем | Сменить задачу (лучше через CLI, см. §2) |
| `sequence_length` | Длина истории (шагов = секунд) | Больше контекст = медленнее. 120 ≈ 2 мин обычно хватает |
| `prediction_horizon` | Горизонт прогноза (шагов) | **Меньше = точнее** (главный урок v003: 10 → R²=0.62) |
| `encoder_hidden_dims`, `temporal_hidden_size`, `temporal_num_layers`, `decoder_hidden_dim` | Размер модели | Больше = потенциально точнее, но дольше и риск переобучения |
| `use_attention` | Внимание в декодере | `True` — даёт качество, замедляет ~20% |
| `bidirectional` | Двунаправленный энкодер | `False` для скорости (вдвое дороже) |
| `batch_size` | Батч | CUDA OOM → уменьшить; быстрый GPU → увеличить |
| `learning_rate` | Скорость обучения | 5e-4 нормально; маленькая модель училась с 1e-3 |
| `num_epochs`, `early_stopping_patience` | Длительность | Ранняя остановка сработает сама |
| `cyclic_encoding` | Углы → sin/cos | Оставить `True` (у legacy-моделей было `False`) |
| `relative_wave_angle` | Отн. угол встречи волны | Оставить `True` — главный физический признак качки |
| `encoder_dropout` / `temporal_dropout` | Регуляризация | Val loss растёт при падении train → увеличить до 0.2 |

**Веса целей** (что важнее в loss) — внутри `Config.get_profile_config()`, ключ `'weights'`:
например `'Roll(градусы)': 2.0` — крен вдвое важнее прочего.

**Профили** (цели):
- `motion_prediction` — качка (6 целей: позиции + скорости)
- `motion_core_prediction` — **только Pitch/Roll/Vertical** (без зашумлённых производных; `profile core` в CLI)
- `rot_prediction` — ROT (поворот)
- `speed_prediction` — SOG (скорость)
- `full_prediction` — всё (8 целей)
- `custom` — свои списки `feature_columns` / `target_columns` прямо в config.py

---

## 2. Переключение профилей и режимов — `config/switch_config.py`

CLI **правит config.py** за вас (regex). После переключения можно открыть файл и проверить.

```bash
python config/switch_config.py              # показать текущую конфигурацию
python config/switch_config.py list         # все профили и режимы

python config/switch_config.py profile core     # motion | core | rot | speed | full
python config/switch_config.py speed fast       # fast | balanced | quality
```

| Режим | seq / horizon / batch | Время (оценка) |
|---|---|---|
| fast | 120 / 20 / 48 | ~1 ч |
| balanced | 180 / 30 / 32 | ~2–3 ч |
| quality | 300 / 60 / 32 | ~6–8 ч |

---

## 3. Обучение — `run_training.py`

```bash
# Обычный запуск (создаст новую версию в models_archive/)
python run_training.py --notes "полный профиль, горизонт 20, базовый прогон"

# Быстрая проверка пайплайна (2 минуты вместо часов)
python run_training.py --data-limit 3000 --max-epochs 2 --notes "smoke"

# Мультисид-эксперимент (запустить 3 раза с разными seed)
python run_training.py --seed 42 --notes "seed 42"
python run_training.py --seed 43 --notes "seed 43"
```

Что происходит автоматически:
1. создаётся папка `models_archive/<run_id>/` (run_id = дата-время-хэш, **не переименовывать!**);
2. пишется `manifest.json` — все условия, данные (sha256), git-коммит, seed;
3. обучение (AdamW + ReduceLROnPlateau + early stopping); в консоли —
   «Top 3 worst variables» (какая цель хуже всех — кандидаты на смену весов/признаков);
4. тест: метрики в **физических единицах** + бейзлайны (persistence/linear) + **skill score**
   (skill > 0 = модель лучше наивного прогноза);
5. строка в `models_archive/experiments.csv`.

---

## 4. Прогноз/проверка — `run_inference.py`

```bash
python run_inference.py                              # интерактивно: спросит модель и параметры
python run_inference.py --run-id production          # модель со статусом production
python run_inference.py --run-id 20251219-120419     # по run_id (можно префикс)
python run_inference.py --run-id latest
python run_inference.py --model-version 3            # legacy-номер (папка v003_*)
python run_inference.py --run-id v003 --mc-samples 30   # + неопределённость (MC-Dropout)
```

Интерактивные вопросы (Enter = значение по умолчанию):
- **History length** (длина истории в шагах): можно НЕ как при обучении — модель Seq2Seq это позволяет. Рекомендация: 30–120
- **Prediction horizon** (горизонт): рекомендация ≤ обучающего (для v003 — ≤10–15)
- **Number of tests**: сколько точек проверить (5)
- **Test mode**: `even` (равномерно) / `manual` (свои индексы) / `random`

Результаты → `results/inference_<run>_<время>/`:
- `prediction_<idx>.png` — график: история + прогноз (синий) + факт (красный) + MAE/RMSE/MAPE
- `predictions.csv` — прогноз/факт/ошибка по каждому шагу (+ `std_*` при MC-Dropout)
- `summary_statistics.txt` — сводка по всем тестам и по каждой цели

---

## 5. Анализ для исследования качки — `scripts/`

### 5.0 Сырые экспорты Transas → модельная схема (конвертер, 2026-10-01)

```bash
# Диагностика: какие каналы распознаны в сыром экспорте (UTF-16 TSV с шапкой)
.venv\Scripts\python.exe -X utf8 scripts\convert_transas_csv.py data\raw\2026-10-01_w-5.csv --list-columns

# Конвертация: фильтр «только нужные колонки» (union всех профилей), без time,
# хвост обрезан до кратности 1000 (безопасная склейка записей в один файл)
.venv\Scripts\python.exe -X utf8 scripts\convert_transas_csv.py data\raw\*.csv --out data\converted --drop-time --align-chunks 1000

# Проверка конверта: схема, время, NaN, обучаемость по всем профилям
.venv\Scripts\python.exe -X utf8 scripts\check_converted.py

# Склейка нескольких записей в один train-файл (кратность чанкам обязательна!)
# и режимная синтетика (B4): автокалибровка по ступеням Wave.Highest
.venv\Scripts\python.exe -X utf8 scripts\generate_regime_synthetic.py --seed 7
```

Ключевые файлы: `data/converted/<имя>_converted.csv` — модельная схема;
`real_w5w6w7_merged.csv` — склейка 3 записей (71000 строк); `synthetic_regime.csv` —
режимная синтетика. Ограничения сырого экспорта: Pitch вырожден (исключён из
профиля `transas_core`), SOG≡0 → подставляется STW, Long/Lat не пишутся —
подробности и чек-лист МТЦ: `docs/TRANSAS_DATA_COLLECTION_PLAN.md` §8.5.

```bash
# Режимы волнения: где модель работает, а где нет
python scripts/evaluate_regimes.py --run-id <run_id> --stride 7
python scripts/evaluate_regimes.py --run-id <run_id> --rows 10000:12499   # свой диапазон

# Непрерывный rolling-прогноз + спектры + фаза качки
python scripts/rolling_eval.py --run-id <run_id> --lead 5 --target "Roll(градусы)"
python scripts/rolling_eval.py --run-id <run_id> --lead 1 --target "Pitch(градусы)"

# Сравнение всех моделей на ОДИНАКОВЫХ окнах (честное сравнение)
python scripts/compare_models.py --all --horizon 10 --num-windows 60
python scripts/compare_models.py --runs v003 v004 --horizon 10
```

Как читать:
- `evaluate_regimes` → ищите режимы, где MAE в разы хуже среднего (сейчас: severe волны >2.5 м, beam seas);
- `rolling_eval` → **skill > 0** лучше persistence; **фазовый сдвиг ~0** = модель не плывёт по фазе;
  пиковые частоты факта и прогноза должны совпадать (период бортовой качки ~11 с);
- `compare_models` → сравнивайте per-target, а не только общий MAE (общий ломается на Velocity.Rolling).

---

## 6. Управление версиями — реестр

```bash
python model_manager.py list                    # таблица всех моделей (+статусы)
python model_manager.py best --metric r2        # лучшая по метрике
python model_manager.py info --version <run_id> # детали конкретной
python model_manager.py compare <id1> <id2>     # сравнение двух

# Сменить статус на production (для inference по --run-id production):
python -c "import json,pathlib,registry; p=pathlib.Path('models_archive/<RUN_ID>/manifest.json'); m=json.load(open(p,encoding='utf-8')); m['status']='production'; json.dump(m,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False); registry.rebuild_registry()"

# Удалить неудачный прогон (папка + строка реестра):
python -c "import registry; registry.delete_run('<RUN_ID>')"
```

`experiments.csv` можно открывать в Excel — это оглавление всех экспериментов.
**Приоритет правды** — `manifest.json` внутри папки запуска; CSV пересобирается из manifests.

---

## 6.1 ОНЛАЙН-СИСТЕМА — режимы A/B (пакет `online/`)

> План: `docs/ONLINE_SYSTEM_PLAN.md`; решения: `docs/ONLINE_DECISIONS.md`;
> журнал сборки: `docs/ONLINE_BUILD_LOG.md`; контракт сенсоров: `docs/ONLINE_API.md`.
> Таймауты/пороги — в одном месте: `config/online.py`.

### Веб-интерфейс (рекомендуется)

```powershell
python -m online.server --port 8765
# открыть http://127.0.0.1:8765 (только локально)
```

**Панель управления проектом** — разделы: Обзор / Обучение (любой профиль +
warm start) / Тест-прогноз / Онлайн (A+B) / Данные (загрузка + генерация
синтетики) / Дообучение / Реестр (promote/delete) / Задачи. У каждой настройки
знак **?** — наведение даёт краткую подсказку, клик — подробную справку
«что произойдёт и зачем».

- Режим A: playback CSV «как сенсор» (1 строка/сек, скорость 1×–100×/max);
- Режим B: live (mock-сенсор: seed, длительность, «уникальные вставки»);
- график: факт + прогноз на 20 с + лента неопределённости MC-Dropout (mean±1.96·std,
  N4/C4) + маркеры; запись серверная — при закрытом браузере сессия продолжается;
- статус «не верить прогнозу» — MC-std выше эмпирического порога (вход СППР;
  пороги per-target — `config/online.py: uncertain_max_std`, значения из
  `results/mc_dropout_n4/`);
- артефакты сессии: `results/online/<id>_{playback,live}/`
  (`session.csv` + маркеры, `forecasts.csv`, `uncertainty.csv`, `summary.txt`,
  `events.log`, `config_snapshot.json`).

### CLI playback (без браузера)

```powershell
python -m online.playback --model <run-id> --csv data/raw/your_data.csv \
       --start-row 4000 --limit 300 --speed max
```

### Дообучение на «уникальных» данных (режим B)

> ⚠ **Инвариант «судно × загрузка» (CONCEPT §3 v1.1):** сегменты сессии для дообучения
> обязаны быть записаны на том же судне и в том же состоянии загрузки (груз/балласт),
> что и исходный дата-сет модели. Смешение загрузок/судов ломает модель —
> записанные сессии нужно атрибутировать и хранить раздельно по этой категории.

```powershell
# маркеры error_anomaly = «модель устойчиво ошибается» → кандидат на дообучение
python -m online.finetune --session results/online/<id>_live --run-id <base_run_id>
# watchdog'и: min 600 строк, max 20 эпох, patience 5, wall-clock 30 мин,
# cooldown 60 мин; валидационный гейт по holdout (см. план §8, ADR-6)
```

### Маркеры (не смешивать!)

| Маркер | Значение | Действие |
|---|---|---|
| `input_anomaly` | вход вне train-распределения (z>4) | оператору: проверить условия/датчик |
| `error_anomaly` | модель устойчиво ошибается (ratio>2.5×базового MAE) | кандидат на дообучение |

### Тесты

```powershell
python tests/test_online.py    # ядро: буфер, watchdog, NaN-гейт, экспорт сегментов
python tests/test_anomaly.py   # детекторы аномалий
python tests/test_live.py      # live-источники (mock/file)
```

---

## 7. Где какие результаты лежат

```
models_archive/
├── experiments.csv                  ← ОГЛАВЛЕНИЕ всех экспериментов
└── <run_id>/
    ├── manifest.json                ← ВСЁ об этом прогоне (условия + результаты)
    ├── training_summary.txt         ← человекочитаемая сводка (метрики, бейзлайны)
    ├── checkpoints/best_model.pt    ← веса модели
    ├── checkpoints/scalers.pkl      ← нормализация
    └── environment.txt              ← версии пакетов
results/
├── article/figs_v2/                 ← рисунки статьи (текущий канон fig1…fig7)
├── sweep_f3/, horizons_f4/          ← sweep и матрица горизонтов (текущий канон)
├── learning_curve_f1/, f1_logs/     ← learning curve + сырые логи Ф1–Ф4
├── regimes/<run>/, rolling/<run>/   ← таблицы по режимам, rolling-анализ прогона
├── online/<id>_{playback,live}/     ← сессии панели (транзиентные)
└── archive/                         ← легаси первой версии
archive/                              ← архив старого пайплайна и тестовых сессий
                                      (перенесено 2026-09-29, ничего не удалено) — archive/README.md
```

---

## 8. Готовые сценарии

### «Проверить, что всё работает» (2 минуты)
```bash
python run_training.py --data-limit 3000 --max-epochs 2 --notes smoke
```

### «Лучшая модель для качки» (основной сценарий)
```bash
python config/switch_config.py profile core      # только Pitch/Roll/Vertical
python config/switch_config.py speed fast
python run_training.py --notes "core, fast, baseline-2"
python run_inference.py --run-id latest
python scripts/rolling_eval.py --run-id latest --lead 5 --target "Roll(градусы)"
python scripts/evaluate_regimes.py --run-id latest
```

### «Максимальное качество» (на ночь / GPU)
```bash
python config/switch_config.py profile motion
python config/switch_config.py speed quality
python run_training.py --notes "motion, quality"
```

### «Мультисид» (доказательность результата)
```bash
python run_training.py --seed 42 --notes "ms42"
python run_training.py --seed 43 --notes "ms43"
python run_training.py --seed 44 --notes "ms44"
python scripts/compare_models.py --runs <id1> <id2> <id3> --horizon 10
```

### «Новые данные»
1. Сырой экспорт Transas → `data/raw/` → конвертер (§5.0) → `data/converted/`.
2. Проверить: `scripts/check_converted.py` (все профили OK, NaN=0).
3. Обучение: `python run_training.py --data-path data/converted/<файл>.csv --profile transas_core`.
4. Склейка нескольких записей — только при кратности 1000 (`--align-chunks`).
Колонки должны включать цели и признаки профиля (см. `check_converted.py`).

---

## 9. Частые проблемы

| Симптом | Причина / решение |
|---|---|
| `No module named torch` | Не активировано `.venv` |
| `CUDA out of memory` | Уменьшить `batch_size` в config.py (48 → 32 → 16) |
| `Missing key(s) in state_dict` на v001 | Чекпоинт архитектуры до bridge-слоёв; модель только архивная (помечена в manifest) |
| `InconsistentVersionWarning` от sklearn | Старые скалеры от sklearn 1.8.0 — работает, предупреждение можно игнорировать |
| Метрики обучения малы, а прогноз плохой | Метрики обучения — в z-score; смотрите физические в training_summary.txt / inference |
| Хочу другой формат данных | `load_data()` в `run_training.py` (sep/encoding) |
| Git: закоммитить новую модель | Модели входят в репозиторий — обычный `git add -A && git commit && git push` |
