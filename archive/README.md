# 📦 ARCHIVE — архив проекта (ничего не удалено, только перенесено)

> Создан аудитом 2026-09-29. Всё содержимое — результаты **старого пайплайна**
> (полный датасет 39 колонок, без КУ ветра) и тестовые сессии панели.
> Актуальный канон: **минимальный датасет** (17 колонок, `your_data_minimal.csv`)
> + профиль `minimal_prediction` — см. `IDEAS.md` §H.
> **Читать только точечно, для истории — не загружать целиком.**

## Структура

| Путь | Что это | Откуда перенесено |
|---|---|---|
| `online_sessions/` | 38 тестовых сессий панели (playback/live, 20260927–20260928) | `results/online/` |
| `results_old_pipeline/article/figs/` | рисунки статьи старого канона fig1…fig7 | `results/article/figs/` |
| `results_old_pipeline/horizons/` | матрица горизонтов A3, старый канон | `results/horizons/` |
| `results_old_pipeline/sweep/` | sweep коэффициентов + мультисид A1, старый канон | `results/sweep/` |
| `results_old_pipeline/learning_curve/` | learning curve E1–E3, старый канон | `results/learning_curve/` |
| `results_old_pipeline/regimes_v003_20251218_221330/` | per-regime оценка модели v003 (первая версия) | `results/regimes/` |
| `results_old_pipeline/rolling_v003_20251218_221330/` | rolling-анализ модели v003 (первая версия) | `results/rolling/` |
| `results_old_pipeline/horizon_sweep.log` | сырой лог прогона горизонтов (7,8 МБ, отладка) | `results/` |
| `results_old_pipeline/panel_test.log` | пустой лог теста панели | `results/` |

## Актуальные результаты (НЕ здесь!)

- рисунки статьи (текущий канон): `results/article/figs_v2/` (fig1…fig7);
- матрица горизонтов (текущий канон): `results/horizons_f4/`;
- sweep (текущий канон): `results/sweep_f3/`;
- learning curve (текущий канон): `results/learning_curve_f1/`;
- сырые логи Ф1–Ф4: `results/f1_logs/`;
- последняя online-сессия (пример работы): `results/online/20260929_053332_playback/`;
- per-regime актуального прогона: `results/regimes/20260929-035843-f1-minimal-k9-s42-3900/`;
- легаси первой версии (inference v003–v007, compare): `results/archive/` — перенесено ранее,
  оставлено на месте.

## Модели старого пайплайна

88 архивных прогонов (E1-исследование, полный датасет) — `models_archive/archive/`
(перенесено ранее, оставлено на месте; в git только манифесты и метрики).
Модели первой версии до manifest-схемы (v001–v004) — там же.
