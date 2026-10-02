# 🗺 MAP.md — карта файлов проекта

> Что делает/показывает каждый файл в каждой папке. Обновляется при аудитах
> (последний — `docs/AUDIT_20261003.md`). Для навигации по смыслу см. `CONTEXT.md`.

## Корень

| Файл | Назначение |
|---|---|
| `run_training.py` | **CLI обучения** — все эксперименты идут через него (профиль, H, K, сиды, extra-train, fixed scaler); пишет manifest в `models_archive/` |
| `run_inference.py` | CLI инференс/тест-прогноз обученной модели |
| `registry.py` | Реестр прогонов: list_runs, манифесты, experiments.csv, promote/delete, sha256 |
| `requirements.txt` | Зависимости Python (venv `.venv`) |
| `AGENTS.md` | Правила работы ИИ-агентов (грузится автоматически) |
| `CONTEXT.md` | Протокол чтения проекта для ИИ/новых участников |
| `CONCEPT.md` | Замысел: инвариант «судно × загрузка», покрытие Ω, СППР |
| `PROJECT_OVERVIEW.md` | Снапшот состояния: что сделано, очередь задач |
| `RESEARCH_LOG.md` | Журнал фактов экспериментов (append-only) — источник правды №2 |
| `IDEAS.md` | Идеи/roadmap с приоритетами (A/B/C…, N1–N13) |
| `README.md` | Лендинг проекта |
| `USAGE.md` | Как запускать обучение/inference/онлайн/панель |
| `MAP.md` | Эта карта |
| `draft-article/*.docx` | Экспорты черновика статьи (v3 + ревизия); канон — `D:/Temprary/research-article/draft-article/` (md, v4) |

## config/

| Файл | Назначение |
|---|---|
| `config.py` | Dataclass Config + `get_profile_config` (features/targets/веса по профилям; канон — `transas_core`) |
| `online.py` | Параметры онлайн-системы (пороги неопределённости MC-Dropout и др.) |
| `switch_config.py` | Переключение конфигураций (вспомогательное) |

## data/

| Файл/папка | Назначение |
|---|---|
| `dataset.py` | VesselDataset: окна seq→horizon, нормализация, сплиты |
| `features.py` | Инжиниринг признаков (циклические углы, относительные углы) |
| `raw/` | Сырые записи: `your_data.csv` («12k», сильное волнение), `2026-10-01_w-5/6/7*.csv` (+ их converted-копии — легаси раскладки), `your_data_minimal.csv` (усечённая схема), `synthetic_*.csv` (генератор, v1), `test.csv` |
| `converted/` | **Канонические датасеты**: `real_w5w6w7_merged.csv` («71k»), конверты w-5/6/7, `synthetic_regime.csv` (режимный генератор B4), `synthetic_bad_calib.csv` (негативный контроль E3TBAD), `test_converted.csv` |
| `online_finetune/` | Сегменты сессий для дообучения (режим B) |

## models/ — архитектура

`vessel_predictor.py` (сборка модели), `encoder.py`, `decoder.py`, `attention.py`, `temporal_block.py`.

## training/ — цикл обучения

`trainer.py` (цикл эпох, patience, чекпоинты), `losses.py` (взвешенные лоссы, roll-веса), `metrics.py` (MAE/R²/skill, физ. единицы), `baselines.py` (persistence для skill).

## online/ — онлайн-система

`server.py` (панель, порт 8765), `playback.py` (CLI/режим A), `engine.py` (прогноз+MC-Dropout), `buffer.py` (окно сенсора), `anomaly.py` (маркеры input/error), `finetune.py` (дообучение по маркерам с гейтами), `api.py`, `sources.py` (CSV/live-источники), `markers.py`, `uncertainty.py`.

## scripts/ — эксперименты и аналитика (активные)

### Мастер-цикл (v2-датасеты)
| Файл | Назначение |
|---|---|
| `run_full_cycle.sh` | **одна команда — весь цикл** (A3.1 → combined → FULL-сидов → A4 → отчёты); маркеры `.done` |
| `cycle_config.sh` | Конфиг цикла: пути датасетов, CYCLE_TAG, сетка K/H/сидов, окно playback |
| `run_cycle_combined.sh` | Этап C: реал+реал C12/C71/FULL (чужой режим в train, свой test) |
| `run_a4_full_seeds.sh` | FULL-сидов 43/44 (v1) или все 3 (v2) |
| `preflight_dataset.py` | **Префлайт нового CSV**: схема, время, NaN, физика, режим, Ω-покрытие → вердикт |
| `omega_density.py` | Зачаток G2/N10: плотность train-условий вокруг test-окон (k-NN в Ω) |

### A3.1 (learning curve / качество от объёма)
| Файл | Назначение |
|---|---|
| `run_a31_12k_core.sh` | Этап 1: 12k, H=20, K=2/4/8 × 3 сида |
| `run_a31_stage2_horizons.sh` | Этап 2: 12k, H=10/30 (18 прогонов) |
| `run_a31_stage3_71k_horizons.sh` | Этап 3: 71k, H=10/30 × K=2/4/8/16 (24 прогона) |
| `a31_learning_curve_report.py` | Сводка этапа 1 (12k vs 71k, H=20) |
| `a31_stage2_horizon_report.py` | Сводка этапов 2+3 (таблица по датасет×горизонт, наклоны) |
| `a31_stage4_powerlaw.py` | Этап 4: степенной закон metric=c·n^b + bootstrap-ДИ по сидам |

### E-цикл (симулятор → реальность, аугментация)
| Файл | Назначение |
|---|---|
| `run_e1prime_batch.sh` | E1T: learning curve на 71k (реал-only) |
| `run_e23prime_batch.sh` | E2T (syn-only) + E3T (реал+syn30k) |
| `run_e3controls_batch.sh` | Контроли: E3TBAD (чужая калибровка) / E3TFULL / E3TDOSE |
| `make_bad_synthetic.py` | Генератор «чужого домена» для негативного контроля |
| `summarize_e3_controls.py` | Сводка E-цикла + проверка зарегистрированных ожиданий |
| `watch_e3controls.sh` | Сторож батча (авто-сводка по завершении) |

### Конвейер данных и генераторы
| Файл | Назначение |
|---|---|
| `convert_transas_csv.py` | Экспорт Transas → канонический CSV (utf-16/tab) |
| `check_converted.py` | Проверка конвертов: схема, 1 Гц, NaN, обучаемость профилей |
| `generate_regime_synthetic.py` | Режимно-зависимый генератор синтетики с автокалибровкой (B4) |
| `make_minimal_dataset.py` | Усечённый датасет/профиль minimal_prediction |
| `analyze_real_data.py`, `analyze_regimes_real.py` | Портрет записей, разбор режимов |

### Отчёты/аналитика (читают манифесты, CPU)
`horizon_report.py`, `f5_horizon_ext_report.py` (кривая по H), `learning_curve_report.py` (Ф1), `sweep.py`/`sweep_report.py` (F3 коэффициенты), `mc_dropout_eval.py` (N4), `rolling_eval.py` (скользящее), `evaluate_regimes.py`, `article_figures.py` / `article_figures_ecycle.py` (фигуры статьи), `a1_finish.py`, `analyze_*.py` (точечные разборы), `param_count_analysis.py`, `fix_stale_runs.py` (ремонт манифестов), `watch_f5ext.sh`/`watch_f5v2.sh` (сторожа), `horizon_sweep.py`.

## scripts/archive/ — одноразовые, НЕ запускать

`check_data.py`, `compare_models.py`, `diagnose_perfomance.py`, `migrate_models_archive.py`, `patch_draft_ru.py`, `__pycache__`.

## tests/

`test_online.py`, `test_live.py`, `test_anomaly.py`, `test_server_fixes.py` (онлайн-система), `test_convert_and_regime.py` (конвертация+генератор), `test_smoke.py` (smoke).

## models_archive/

Один каталог на прогон: `manifest.json` (источник правды по прогону), `checkpoints/` (best_model.pt + scalers.pkl — не в git), `environment.txt`, `training_summary.txt`. Сводка — `experiments.csv` (реестр). `archive/models_f5_protocol_v1/` — устаревший первый батч Ф5 (сплит с ошибкой).

## results/

| Папка | Содержимое |
|---|---|
| `a31_learning_curve/` | **Канон A3.1**: сводные этапов 1–4 (`a31_matched_*`, `a31_stage2_*`, `a31_stage4_powerlaw.*`), логи батчей |
| `a31_learning_cycle/` | Мастер-цикл: маркеры `.done`, логи combined, отчёты `report_stage23/4.txt` |
| `learning_curve_f1/` | Канон Ф1 (learning curve 12k, старая панель) |
| `sweep_f3/` | Канон F3 (влияние коэффициентов) |
| `horizons_f4/` | Канон Ф4+Ф5-ext (кривая по H, 5–120 с) |
| `learning_curve_e1t/e2t/e3t/`, `real_e1prime/` | E-цикл: кривые и логи |
| `e3_controls/` | Контрольные серии E-цикла + сводка ожиданий |
| `mc_dropout_n4/` | Валидация неопределённости (N4/C4) |
| `analysis_real_data/` | Портрет записей |
| `analysis_course_design/`, `real_w5w7_eda/`, `regimes/`, `rolling/` | Точечные анализы режимов/курса/скользящее |
| `regime_generator_validation.txt` | Валидация генератора B4 |
| `article/figs_v2/` | **Канон фигур статьи** |
| `online/` | Сессии панели/playback + `a4_playback_multiseed_report.txt` |
| `preflight/` | Отчёты префлайта датасетов |
| `omega_density/` | Расчёты плотности Ω (txt + npy) |
| `f1_logs/`, `snapshots/` | Логи прогонов Ф1, снапшоты панели |
| `archive/` | Легаси первой версии (не трогать) |

## docs/

| Файл | Назначение |
|---|---|
| `ARTICLE_PLAN.md`, `ARTICLE_DRAFT_RU.md` | План и черновик статьи |
| `DATASET_REGISTRY.md` | Карточки датасетов + правила добавления новых |
| `AUDIT_20261003.md` | Последний аудит структуры/дубликатов |
| `MINIMAL_DATASET.md` | Обоснование минимального датасета |
| `TRANSAS_DATA_COLLECTION_PLAN.md` | План сбора записей (сетка Ω, маркеры, экспорт) |
| `ONLINE_SYSTEM_PLAN.md` / `ONLINE_BUILD_LOG.md` / `ONLINE_DECISIONS.md` | План/журнал/решения онлайн-системы |
| `ONLINE_API.md` | Контракт live-источника (режим B) |
| `Oil tanker 70k loaded.pdf`, `Wheel-house poster ...pdf` | Паспорт судна записи |
| `enviroment 2.envtmpl` | Шаблон окружения записи (бинарник Transas) |
| `TASKS_current.txt` / `TASKS_history.txt` | Текущие/выполненные задачи заказчика |
| `1.csv`, `2.csv`, `3.csv` | Вспомогательные данные (клеймы записи) |
| `history/` | История v001–v004 + легаси-заметки (точечно, целиком не читать) |

## Служебные (в git не входят / не редактировать вручную)

`.venv/`, `.git/`, `.idea/`, `.obsidian/`, `.pi/`, `.pytest_cache/`, `__pycache__/`.
