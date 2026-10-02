# 📦 Реестр датасетов проекта

> Карточка на каждую запись. При поступлении нового файла:
> 1) `python scripts/preflight_dataset.py <файл> --tag <vN> --save` → отчёт в `results/preflight/`;
> 2) вставить сюда карточку с итогом префлайта;
> 3) только после этого запускать обучение.

## your_data.csv — «12k», сильное волнение

| Поле | Значение |
|---|---|
| Путь | `data/raw/your_data.csv` |
| Объём | 12 499 строк, 1 Гц (~3.5 ч) |
| Кодировка/разделитель | utf-16 / TAB |
| Режим | сильное волнение: Roll std ≈ 0.83°, Wave.Highest mean ~? |
| Ω-покрытие | узкое (см. `results/omega_density/`) |
| Префлайт | ✅ `results/preflight/your_data.txt` (0 блокеров; нет Wind.speed — не блокер) |
| Особенности | броски Velocity.Rolling до ±660°/мин — реальные, не дефект |
| Использована в | Ф1/Ф3/Ф4/Ф5, A3.1 этапы 1–2, E3TBAD (чужой домен) |

## real_w5w6w7_merged.csv — «71k», спокойные режимы (w-5/6/7)

| Поле | Значение |
|---|---|
| Путь | `data/converted/real_w5w6w7_merged.csv` |
| Объём | 71 000 строк, 1 Гц (~19.7 ч, 3 записи merged) |
| Кодировка/разделитель | utf-16 / TAB |
| Режим | спокойный: Roll std ≈ 0.26°, Wave.Highest mean 3.39 м |
| Ω-покрытие | широкое: 21 закрытая ячейка сетки волнение×КУ×скорость, 0 дыр |
| Префлайт | ✅ `results/preflight/real_w5w6w7_merged.txt` (0 блокеров) |
| Особенности | нет колонки Wind.speed (не блокер, вне transas_core) |
| Использована в | E1T/E2T/E3T + контроли, A3.1 этап 3, A4, C71 (combined) |

## Комбинированные прогон-пары

| Тег | Состав | Статус |
|---|---|---|
| C12 | train = 12k(K) + 71k(extra), test = 12k | 🔄 батч идёт (`batch_combined.log`) |
| C71 | train = 71k(K) + 12k(extra), test = 71k | 🔄 батч идёт |
| C12FULL/C71FULL | полные объёмы обоих файлов | 🔄 батч идёт |

## Правила добавления нового датасета (v2+)

1. Префлайт → карточка здесь → отчёт в `results/preflight/`.
2. `scripts/cycle_config.sh`: обновить `DS_12K`/`DS_71K`, поставить `CYCLE_TAG=v2`.
3. `bash scripts/run_full_cycle.sh` — весь цикл (A3.1 + combined + A4 + отчёты);
   маркеры `.done` кладутся в `results/a31_learning_cycle_v2/`, старые манифесты не смешиваются.
4. Результаты: `RESEARCH_LOG.md` (append) + сводки `results/a31_learning_cycle_v2/`.
