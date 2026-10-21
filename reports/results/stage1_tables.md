### Релевантность (test)

| модель | all:listen_plus:ndcg@10 | all:listen_plus:recall@10 | all:listen_plus:map@10 | all:listen_plus:ndcg@100 | all:listen_plus:recall@100 | new:listen_plus:ndcg@10 | new:listen_plus:recall@100 | all:like:ndcg@100 | all:like:recall@100 |
|---|---|---|---|---|---|---|---|---|---|
| MostPop | 0.0253 | 0.0258 | 0.0110 | 0.0342 | 0.0560 | 0.0224 | 0.0562 | 0.0281 | 0.0718 |
| PersonalRepeat | 0.2033 | 0.2028 | 0.1452 | 0.2325 | 0.3124 | 0.0224 | 0.0562 | 0.0306 | 0.1071 |
| GPT2 | 0.1123 | 0.1098 | 0.0654 | 0.1389 | 0.1951 | 0.0389 | 0.1286 | 0.0324 | 0.1002 |

### Свойства списков, протокол all

| модель | coverage@10 | coverage@100 | gini@10 | pair_jaccard@10 | in_history@10 | log10_train_count@10 | track_length_sec@10 | unique_artists@10 |
|---|---|---|---|---|---|---|---|---|
| MostPop | 0.000 | 0.001 | 1.000 | 1.000 | 0.209 | 3.409 | 164.500 | 9.000 |
| PersonalRepeat | 0.135 | 0.534 | 0.922 | 0.001 | 0.999 | 3.005 | 200.116 | 8.437 |
| GPT2 | 0.109 | 0.372 | 0.944 | 0.001 | 0.814 | 3.104 | 203.388 | 7.540 |

### Свойства списков, протокол new

| модель | coverage@10 | coverage@100 | gini@10 | pair_jaccard@10 | in_history@10 | log10_train_count@10 | track_length_sec@10 | unique_artists@10 |
|---|---|---|---|---|---|---|---|---|
| MostPop | 0.000 | 0.002 | 1.000 | 0.611 | 0.000 | 3.227 | 205.579 | 8.641 |
| PersonalRepeat | 0.000 | 0.002 | 1.000 | 0.611 | 0.000 | 3.227 | 205.579 | 8.641 |
| GPT2 | 0.116 | 0.379 | 0.936 | 0.001 | 0.000 | 3.031 | 203.971 | 8.413 |
