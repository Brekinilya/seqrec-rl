# RL-дообучение последовательной рекомендательной модели

Исследуем, как RL-дообучение GPT-2 рекомендера на награду от reward model меняет его поведение. Смотрим, растёт ли награда, сохраняется ли релевантность, какие хаки появляются и можно ли сформулировать долгосрочную цель. Данные — [Yambda-50M](https://huggingface.co/datasets/yandex/yambda) (Яндекс Музыка).

## Этапы

| этап | что | статус |
|---|---|---|
| 0 | данные, разбиение по времени, EDA — [`reports/eda.md`](reports/eda.md) | готово |
| 1 | базовая GPT-2 модель next-item prediction и бейзлайны | — |
| 2 | reward model (доля дослушивания, лайки, долгосрочные сигналы) | — |
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

## Структура

```
src/seqrec_rl/   загрузка данных, разбиение, стиль графиков
scripts/         запускаемые шаги пайплайна
reports/         отчёты, графики, числа EDA
data/            скачанные данные (не в git)
```
