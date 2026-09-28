# 🤖 CONTEXT.md — протокол чтения для ИИ и новых участников

> Цель: стартовать сессию с минимальным контекстом. **НЕ читайте проект целиком** —
> читайте по карте ниже. Полные документы существуют, но открываются точечно.

## Минимальный стартовый набор (всегда, ~3 файла, ~250 строк суммарно)

1. `CONCEPT.md` — замысел (что строим, инвариант загрузки, покрытие Ω) — **читать целиком**
2. `PROJECT_OVERVIEW.md` — снапшот состояния (компактный) — **читать целиком**
3. `RESEARCH_LOG.md` — **только §5 (результаты), §6 (ошибки), §7 (выводы/следующие шаги)**

## Точечное чтение (по необходимости)

| Вопрос / задача | Что открыть |
|---|---|
| Быстрый старт/что где нажимать | `README.md` (лендинг) |
| Как запустить обучение/inference/анализ | `USAGE.md` (целиком, ~200 строк) |
| Параметры модели/профиля | `config/config.py` (только поля dataclass + `get_profile_config`) |
| Как устроена архитектура | `models/vessel_predictor.py` (+ `encoder.py`, `decoder.py`, `attention.py` при необходимости) |
| История версий v001–v004, старые метрики | `docs/history/PROJECT_OVERVIEW_v1_20260927.md` §7–9 (точечно, целиком НЕ читать) |
| Какая версия модели выбрать | `models_archive/experiments.csv` (CSV) или панель «Реестр» (кнопка В production) |
| Онлайн-система (playback/live/аномалии/дообучение) | `docs/ONLINE_SYSTEM_PLAN.md` (план §10 — прогресс), `docs/ONLINE_BUILD_LOG.md` (журнал сборки), `docs/ONLINE_DECISIONS.md` (решения ±) |
| Контракт live-источника данных (режим B) | `docs/ONLINE_API.md` |
| Куда идём дальше (идеи/roadmap LSTM) | `IDEAS.md` (статус D1–D4, приоритеты) |
| Конкретный прогон | `models_archive/<run_id>/manifest.json` (источник правды) |
| Статьи по теме — где что лежит | см. блок «Научные работы» ниже |
| Черновик статьи | `/d/Temprary/research-article/draft-article/` |

## Раскладка файлов (после cleanup 2026-09-28)

- `!version_1` и `README/` удалены; корень — только CLI + `.md`-документы + `requirements.txt`;
- `model_manager.py` → `docs/history/legacy_notes/` (легаси, заменён `registry.py`);
- одноразовые скрипты → `scripts/archive/` (выполнены, не запускать);
- `results/`: рабочие папки без run-id-суффиксов (`regimes/<run>/`, `rolling/<run>/`),
  легаси-вывод — `results/archive/`;
- `docs/TASKS_history.txt` — выполненные задачи заказчика (бывш. `Tasks fm user.txt`);
- чекпоинты моделей (`best_model.pt`/`scalers.pkl`) НЕ в git — восстанавливаются
  перезапуском обучения по manifest; в git только манифесты и метрики.

## Научные работы (библиотека research-article, 2026-09-28)

- Скачиваются статьи/книги → `D:\Temprary\research-article\articles\`
  (группировка по кластерам: autonomous-control, general, ship-motion-prediction,
  trajectory-prediction, safety-extreme-conditions, storm-routing-weather,
  ship-maneuvering-model, review-guidelines; новые — в `articles/_inbox/`).
- Мастер-таблица после группировок → `D:\Temprary\research-article\notes\library.md`
  (карточки ⬜/🤖/✅, статусы анализа). ⚠ В таблице отображены ещё НЕ все статьи/книги —
  догружается по мере разборки `_inbox`.
- Анализы (карточки-саммари по статьям) → `D:\Temprary\research-article\analysis\`
  (`<кластер>__<короткое имя>.md`; глубокие карточки — в `articles/<кластер>/<имя>/`).
- Тематические подборки новостей: `notes/web-lstm-articles.md`, `notes/web-sensor-articles.md`.
- Позже появится отдельный файл с самыми полезными статьями — будет добавлен сюда по
  указанию заказчика.
- Пункт F2 (словарь состояний волнения) — вернуться позже; литературная база для него —
  эта же библиотека (см. IDEAS.md §F).

## Правила экономии контекста

- Не читать `docs/history/` без явной необходимости; там ≥ 450 строк истории.
- Не перечитывать большие CSV/PNG; для результатов есть готовые сводки
  (`results/learning_curve/learning_curve_report.txt`, `results/analysis_real_data/report.txt`).
- Логи обучения не читать из консоли истории — читать `training_summary.txt` прогона.
- При длинных сессиях возвращаться к этому файлу, а не к «памяти» о прочитанном.

## Что считается источником правды (иерархия)

1. `CONCEPT.md` — замысел (меняется только явно, с записью в §9)
2. `RESEARCH_LOG.md` — факты экспериментов (append-only по возможности)
3. `models_archive/<run_id>/manifest.json` — факты конкретного прогона
4. Код — факты поведения; `PROJECT_OVERVIEW.md` — снапшот состояния
