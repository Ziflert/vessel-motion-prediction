# AGENTS.md — правила работы ИИ-агентов в проекте

> Этот файл pi загружает в системный промпт автоматически в каждой сессии.
> Он НЕ дублирует `CONTEXT.md` (навигация по файлам) и `CONCEPT.md` (замысел) —
> здесь только протокол поведения и правила использования скилов.

## 1. Навигация (первое действие в сессии)

- **Всегда начинать с `CONTEXT.md`** — там карта чтения: минимальный стартовый набор
  (~3 файла) и точечное чтение по вопросам. **Проект целиком НЕ читать.**
- Замысел проекта: `CONCEPT.md` (инвариант «судно × загрузка», покрытие Ω, СППР).
- Снапшот состояния: `PROJECT_OVERVIEW.md`. Открытые задачи: `PROJECT_OVERVIEW.md` §«Открытые задачи» + `IDEAS.md` §E.
- Не читать без нужды: `docs/history/`, `RESEARCH_LOG.md` §1–4, большие CSV/PNG,
  логи из консоли. Готовые сводки: `results/learning_curve_f1/learning_curve_report.txt`,
  `results/analysis_real_data/report.txt`, `models_archive/<run_id>/training_summary.txt`.

## 2. Источники правды (иерархия, спорные моменты решаются по ней)

1. `CONCEPT.md` — замысел (меняется только явно, с записью в §9)
2. `RESEARCH_LOG.md` — факты экспериментов (append-only)
3. `models_archive/<run_id>/manifest.json` — факты конкретного прогона
4. Код — факты поведения; `PROJECT_OVERVIEW.md` — снапшот состояния

## 3. Скилы (краткая карта; полная — в скиле `vessel-project`)

Проект — data-driven ML (LSTM/PyTorch, качка судна) + научная статья. Использовать
научные скилы **детерминированно** при триггере; при сомнении —
`/skill:vessel-project` (полная карта триггеров и чек-листы задач).

| Триггер | Скил |
|---|---|
| Новый эксперимент / свип / learning curve / план | `experimental-design`, `hypothesis-generation` |
| Анализ данных / обработка CSV / профили | `exploratory-data-analysis`, `statistical-analysis` |
| Статистика: тесты, регрессия, степенные законы, доводы | `statistical-analysis` (+ `statsmodels` для низкоуровневых API) |
| Sanity check чисел (MAE, R², периоды, единицы) | `uncertainty-and-units` |
| Рисунки статьи (`scripts/article_figures.py`), графики | `scientific-visualization` |
| Правка/создание научных текстов, отчётов | `scientific-writing` |
| Поиск статей / DOI / открытый доступ | `paper-lookup`, `research-lookup` |
| Библиография, BibTeX, проверка цитат | `citation-management` |
| Ревью черновика статьи / рукописи | `peer-review`, `scientific-critical-thinking` |
| Диаграммы в docs (архитектура, пайплайны) | `markdown-mermaid-writing` |
| Постер по статье | `latex-posters` или `pptx-posters` |
| Планирование объёма данных / мощности | `statistical-power` |

**Запрещено вызывать** (вводят в заблуждение, к проекту не относятся): `biopython`,
`adaptyv`, `benchling-integration`, `cobrapy`, `ginkgo-cloud-lab`, `tamarind`,
`diffdock`, `rdkit`/`medchem`/`molfeat`, квантовые (`qiskit`/`cirq`/`pennylane`),
нейровиз (`bids`, `neuropixels-analysis`), wet-lab/химия/докинг — в проекте нет
никаких wet-lab задач. Скилы про LSTM-general (`aeon`, `pyhealth`, `torch-geometric`)
не использовать без явной причины — пайплайн проекта свой (`run_training.py`).

## 4. Правила работы

- **Экономия контекста** — см. `CONTEXT.md` §«Правила экономии контекста». При длинных
  сессиях возвращаться к `CONTEXT.md`, а не к «памяти» о прочитанном.
- **Результаты писать по назначению**: эксперименты → `RESEARCH_LOG.md` (append),
  факты прогонов → `manifest.json`, идеи → `IDEAS.md`, снапшот → `PROJECT_OVERVIEW.md`.
- Чекпоинты (`best_model.pt`, `scalers.pkl`) НЕ в git — восстанавливаются перезапуском
  обучения по manifest; в git только манифесты и метрики.
- `scripts/archive/` — одноразовые выполненные скрипты, **не запускать**.
- Перед правками кода/документов — читать файл, править точно; не переписывать целиком
  без необходимости.
- Windows-окружение: Python 3.13, venv `.venv`; команды — `.venv\Scripts\python.exe`.
- Команды запуска (обучение/inference/онлайн/панель) — `USAGE.md`.

## 5. Чек-лист типовых задач

- **Новый эксперимент** → `IDEAS.md` (приоритет) → скил `experimental-design` +
  `hypothesis-generation` → прогон по `USAGE.md` → результат в `RESEARCH_LOG.md` +
  manifest.
- **Правка статьи** → `docs/ARTICLE_PLAN.md` (структура) → черновик
  `docs/ARTICLE_DRAFT_RU.md` (или `/d/Temprary/research-article/draft-article/`) →
  скилы `scientific-writing` (+ `peer-review` для ревизии).
- **Поиск литературы** → скилы `paper-lookup`/`research-lookup` → файлы в
  `D:\Temprary\research-article\articles\_inbox\` (карта — `CONTEXT.md` §«Научные работы»).
- **Анализ данных** → скил `exploratory-data-analysis` → скрипт в `scripts/`
  (не в `scripts/archive/`) → сводка + запись в `RESEARCH_LOG.md`.
- **Проверка чисел** → скил `uncertainty-and-units` (физические единицы, порядки
  величин, периоды качки).
- **После правки скила** → `/reload` в активной сессии pi.
