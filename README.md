# RL-дообучение последовательной рекомендательной модели

Исследуем, как RL-дообучение GPT-2 рекомендера на награду от reward model меняет его поведение. Смотрим, растёт ли награда, сохраняется ли релевантность, какие хаки появляются и можно ли сформулировать долгосрочную цель. Данные — [Yambda-50M](https://huggingface.co/datasets/yandex/yambda) (Яндекс Музыка).

## Этапы

| этап | что | статус |
|---|---|---|
| 0 | данные, разбиение по времени, EDA — [`reports/eda.md`](reports/eda.md) | готово |
| 1 | базовая GPT-2 модель next-item prediction и бейзлайны — [`reports/stage1.md`](reports/stage1.md) | готово |
| 2 | reward model (доля дослушивания, лайки, долгосрочные сигналы) — [`reports/stage2.md`](reports/stage2.md) | готово |
| 3 | RL-дообучение (GRPO, точная KL-оптимальная политика, DPO) | — |
| 4 | генерация на несколько шагов | — |
| 5 | итоговый отчёт | — |

## Воспроизведение

Нужен Python 3.12 и GPU с CUDA 12.x.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

Данные (~400 МБ) скачиваются в `data/raw/`:

```bash
.venv/Scripts/python scripts/download_data.py
```

EDA (около 2 минут на машине с 16 ГБ RAM):

```bash
.venv/Scripts/python scripts/eda.py
```

Этап 1. Словарь и токенизированный лог (~1 мин), бейзлайны (~4 мин), обучение GPT-2 (~2.5 ч на RTX 3070 Laptop 8 ГБ), оценка и таблицы:

```bash
.venv/Scripts/python scripts/prepare_data.py
.venv/Scripts/python scripts/baselines.py
.venv/Scripts/python scripts/train_base.py --out checkpoints/base
.venv/Scripts/python scripts/evaluate_model.py --ckpt checkpoints/base/best.pt --name GPT2 --parts val test
.venv/Scripts/python scripts/report_stage1.py --models GPT2
```

Если обучение прервалось, его можно продолжить с последнего чекпойнта: `--resume checkpoints/base/last.pt`.

Этап 2. Признаки и метки отклика (~1.5 мин), ансамбль reward model (4 модели, ~35 мин; скрипт можно перезапускать — готовые модели пропускаются, прерванные продолжаются), оценка с бейзлайнами (~10 мин), жадные списки по reward model (~3 мин), таблицы:

```bash
.venv/Scripts/python scripts/prepare_rm_data.py
powershell -ExecutionPolicy Bypass -File scripts/train_rm_ensemble.ps1
.venv/Scripts/python scripts/evaluate_rm.py
.venv/Scripts/python scripts/rm_preferences.py
.venv/Scripts/python scripts/report_stage2.py
```

Обучение на ноутбуке стоит запускать от сети: от батареи GPU ограничен ~30 Вт, и шаг идёт в 7 раз медленнее.

Тесты (нужен GPU):

```bash
.venv/Scripts/python -m pytest tests
```

## Структура

```
src/seqrec_rl/   данные и разбиение, модель и лосс, протокол оценки, бейзлайны,
                 reward model и её метки, бейзлайны отклика, стиль графиков
scripts/         запускаемые шаги пайплайна
tests/           проверки лосса и нарезки окон
reports/         отчёты по этапам, графики, результаты в JSON
data/            скачанные и подготовленные данные (не в git)
checkpoints/     веса моделей (не в git)
runs/            логи и сохранённые выдачи (не в git)
```
