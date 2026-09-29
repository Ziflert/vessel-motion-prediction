---
name: vessel-project
description: Правила и карты для работы в проекте LSTM-Vessel_Motion_Prediction (прогноз качки судна, Seq2Seq/PyTorch) — запуск экспериментов, анализ данных, свипы и learning curves, онлайн-система, научная статья, поиск литературы, ревью. Используй при любой задаче в этом проекте: новый эксперимент, правка статьи, анализ данных, вопрос о модели/данных, коммит.
---

# 🚢 vessel-project — рабочие карты проекта LSTM-Vessel_Motion_Prediction

Проект: data-driven прогноз качки судна. **Seq2Seq (MLP → LSTM-энкодер 128×2 →
attention → LSTM-декодер 96)** на PyTorch. Вход — окно истории бортовых датчиков
(1 Гц), выход — прогноз на 20 шагов по 8 целям. Плюс онлайн-система (playback/live,
аномалии, дообучение) и веб-панель. Научная статья — в подготовке.

## Первое действие: навигация

1. `CONTEXT.md` — карта чтения проекта (что открывать по какому вопросу). Проект
   целиком НЕ читать.
2. `CONCEPT.md` — замысел (инвариант «судно × загрузка», покрытие Ω, СППР) — целиком.
3. `PROJECT_OVERVIEW.md` — снапшот состояния и открытые задачи — целиком.
4. `RESEARCH_LOG.md` — только §5 (результаты), §6 (ошибки), §7 (выводы).

## Критические факты о данных (до любых экспериментов)

- `data/raw/your_data.csv` — сценарий нарастающего шторма (тренажёр Transas);
  Wave.Highest — рампа/сатурация (49 ступеней 1.48→10.8 м, НЕ измерение);
  Wave.direction ≡ 180°, Wind.direction ≡ 0° — задуманный дизайн, погода задавалась
  константами, КУ менялся манёврами курса; погода применяется ОТНОСИТЕЛЬНО судна.
- Спокойный регион 0–4000 НЕ «мёртвый»: там реальные манёвры (14 эпизодов,
  gain «руль→ROT» 0.421 против 0.284 в шторме). Исследовательский регион — ПОЛНАЯ
  запись (Ф0.5 ✅).
- Период качки дрейфует со штормом: Roll T ≈ 8.3→16 с.
- Зависимость «условия → качка» ЕСТЬ: corr(std Roll, h_s)=+0.56, corr(std Pitch, h_s)=+0.74
  (амплитудная мера по 49 состояниям; мгновенная corr=0.00 обманывала — фаза доминирует).
- Контур «руль→ROT→курс» честный: corr(ROT, dCourse)=+0.985.
- **Инвариант «судно × загрузка»**: нельзя смешивать записи разных состояний загрузки
  и тем более разных судов в одном дата-сете (CONCEPT §3).
- Канала Wind.speed в записи НЕТ — добавить при следующем сборе (план v1.1).

## Карта скилов по триггерам (детально)

### Исследование и эксперименты

| Триггер | Скил | Примечание |
|---|---|---|
| Новый эксперимент, свип, learning curve, план исследования | `experimental-design` | дизайн ДО сбора данных: рандомизация, сиды, блоки |
| Новая гипотеза / объяснение наблюдений | `hypothesis-generation` | формулировка в `IDEAS.md`, не «в чате» |
| Сколько данных нужно / обоснование размера выборки | `statistical-power` | для B1 (сбор Transas), Φ0.5, learning curve |
| Планирование множества прогонов / multisid | `experimental-design` | одиночный сид переоценивает эффекты (§5.5 лога) |

### Статистика и анализ данных

| Триггер | Скил | Примечание |
|---|---|---|
| Разведочный анализ CSV, профили данных | `exploratory-data-analysis` | bounded, локально |
| Сравнение групп, тесты, регрессия | `statistical-analysis` | + `statsmodels` для коэффициентов/диагностики |
| Степенные законы, подгонка MAE ~ a·n^b | `statistical-analysis` | как в §5.1 лога |
| Sanity check чисел: MAE, R², периоды качки, единицы | `uncertainty-and-units` | физические единицы, порядки величин |
| Обработка больших CSV | `polars` | 12 499 строк влезают в pandas; polars — для большего |

### Визуализация и графики

| Триггер | Скил | Примечание |
|---|---|---|
| Рисунки статьи (`scripts/article_figures.py`), fig1–fig7 | `scientific-visualization` | аудит правдивости, 300 dpi, PDF+PNG |
| Быстрый статистический график | `seaborn` | для исследований, не для статьи |
| Диаграммы в docs (архитектура, пайплайны, онлн) | `markdown-mermaid-writing` | mermaid как стандарт docs |
| Постер по статье | `latex-posters` / `pptx-posters` | по формату заказчика |

### Научная статья

| Триггер | Скил | Примечание |
|---|---|---|
| Правка/создание разделов статьи | `scientific-writing` | черновик `docs/ARTICLE_DRAFT_RU.md` |
| Ревью черновика / ревизор (N3) | `peer-review` + `scientific-critical-thinking` | план ревизии — по `docs/ARTICLE_PLAN.md` |
| Поиск статей / DOI / открытый доступ | `paper-lookup`, `research-lookup` | файлы → `D:\Temprary\research-article\articles\_inbox\` |
| Библиография, BibTeX, проверка цитат | `citation-management` | проверка перед добавлением |
| Структура/шаблон статьи | `venue-templates` | по журналу/конференции |

### Онлн-система и панель

| Триггер | Скил | Примечание |
|---|---|---|
| Онлайн-система (playback/live/аномалии/дообучение) | — (документы проекта) | `docs/ONLINE_SYSTEM_PLAN.md`, `ONLINE_DECISIONS.md`, `ONLINE_BUILD_LOG.md` |
| Контракт live-источника (режим B) | — | `docs/ONLINE_API.md` |

**Запрещено вызывать** (вводят в заблуждение): `biopython`, `adaptyv`,
`benchling-integration`, `cobrapy`, `ginkgo-cloud-lab`, `tamarind`, `diffdock`,
`rdkit`/`medchem`/`molfeat`, квантовые (`qiskit`/`cirq`/`pennylane`), нейровиз
(`bids`, `neuropixels-analysis`), wet-lab/химия/докинг. Скилы LSTM-general
(`aeon`, `pyhealth`, `torch-geometric`) — без явной причины не использовать:
пайплайн проекта свой (`run_training.py`, `training/`).

## Чек-листы типовых задач

### Новый эксперимент
1. `IDEAS.md` (приоритет §E) → уточнить гипотезу (скил `hypothesis-generation`).
2. Дизайн (скил `experimental-design`): сида ≥3, фиксированный тест, фиксированный
   scaler (уроки §5.1, §5.5).
3. Прогон по `USAGE.md` (CLI `run_training.py` / `scripts/sweep.py`).
4. Результат → `RESEARCH_LOG.md` (append, §5) + `models_archive/<run_id>/manifest.json`.
5. Обновить `PROJECT_OVERVIEW.md` (снапшот) и `IDEAS.md` (статус).

### Правка статьи
1. `docs/ARTICLE_PLAN.md` — структура, файлы, источники.
2. Черновик: `docs/ARTICLE_DRAFT_RU.md` или `/d/Temprary/research-article/draft-article/`.
3. Скилы: `scientific-writing` (текст) + `citation-management` (цитаты).
4. Для ревизии (N3): `peer-review` + `scientific-critical-thinking`.
5. Рисунки: `scripts/article_figures.py` → аудит скилом `scientific-visualization`.

### Поиск литературы
1. Скилы `paper-lookup`/`research-lookup`.
2. Файлы → `D:\Temprary\research-article\articles\_inbox\` (карта — `CONTEXT.md`
   §«Научные работы»; кластеры: ship-motion-prediction, trajectory-prediction,
   safety-extreme-conditions и др.).
3. Мастер-таблица → `D:\Temprary\research-article\notes\library.md`.

### Анализ данных
1. Скил `exploratory-data-analysis` (профиль данных).
2. Скрипт в `scripts/` (НЕ в `scripts/archive/` — там выполненные).
3. Сводка + запись в `RESEARCH_LOG.md`; рисунки по необходимости.

### Обучение / inference / панель
1. Команды — `USAGE.md` (целиком).
2. Windows: `.venv\Scripts\python.exe`; веб-панель: `python -m online.server`
   → http://127.0.0.1:8765.
3. Выбор версии модели: `models_archive/experiments.csv` или панель «Реестр».

## Источники правды

1. `CONCEPT.md` — замысел (меняется только явно, запись в §9)
2. `RESEARCH_LOG.md` — факты экспериментов (append-only)
3. `models_archive/<run_id>/manifest.json` — факты прогона
4. Код — поведение; `PROJECT_OVERVIEW.md` — снапшот

## Экономия контекста

- Не читать `docs/history/`, `RESEARCH_LOG.md` §1–4, большие CSV/PNG, логи консоли.
- Готовые сводки: `results/learning_curve_f1/learning_curve_report.txt`,
  `results/analysis_real_data/report.txt`, `<run>/training_summary.txt`.
- При длинных сессиях возвращаться к `CONTEXT.md`, а не к «памяти».
