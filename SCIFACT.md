# SciFact: второй датасет (ветка `scifact`)

Зачем: на HotpotQA дорогая модель (GPT-OSS-120B) не лучше BERT-реранкера. Проверяем, будет ли иначе на научных текстах,
где BERT менее «свой», а нужен один подходящий абстракт, а не цепочка из двух текстов.
Это второй этап. Он не заменяет HotpotQA и не должен мешать ему. Сначала HotpotQA, потом это.

## Что отличается от HotpotQA

- Вопрос здесь — научное утверждение (claim). Нужный текст — абстракт статьи, которая его подтверждает или опровергает.
- Ответов нет, генератор не нужен. **Команды с `--answers` для SciFact не запускаем.**
- Выборка: 50 проверочных + 150 тестовых утверждений, случайные непересекающиеся подмножества 300 утверждений BEIR test.
  Размеры взяты из расчёта на лимит: абстракты длинные, один запрос к 120B ≈ 4,4 тысячи токенов, то есть ~45 запросов в сутки на ключ.
- Всё делается тем же кодом, но с другим конфигом: `--config config.scifact.toml`. Результаты лежат в `runs/scifact`,
  результаты HotpotQA (`runs/main`) не затрагиваются. `data/` общий: кэш ответов и локальный учёт лимита работают вместе.
- Инструкция для 120B другая (`rerank_instruction` в конфиге): ранжировать по тому, насколько текст подтверждает или опровергает утверждение.
  Размер текста в запросе тот же (192 токена на текст, как в HotpotQA).

## Что делать (Коля)

Лимит тратится только на шаги 6–8. Шаги 1–5 работают локально и бесплатно.

```bash
git fetch origin && git checkout scifact && git pull
source .venv/bin/activate && pip install -e .
# ключ Groq (бесплатный, без карты) должен быть задан, как для HotpotQA: export GROQ_API_KEY=...   (или в .env)

rerank --config config.scifact.toml prepare
rerank --config config.scifact.toml retrieve                          # поиск по 5183 текстам; на моём 4-ядерном процессоре ~30 минут
rerank --config config.scifact.toml run --split validation --method bert
rerank --config config.scifact.toml run --split test --method bert
rerank --config config.scifact.toml evaluate --split validation --method dense   # без API
rerank --config config.scifact.toml evaluate --split validation --method bert    # без API
```

Эти числа у тебя должны получиться такими же, как у меня (см. ниже). Если отличаются, напиши мне, не запускай дорогую модель.

## Что уже известно без дорогой модели

Я прогнал шаги `prepare`, `retrieve` и BERT на этих же 50 + 150 утверждениях (бесплатно, на процессоре). Измерено, nDCG@10:

| Выборка | Только поиск (dense) | + BERT | «Идеальная перестановка» топ-20 |
|---|---|---|---|
| проверочные, 50 | 0,787 | 0,774 | 0,940 |
| тестовые, 150 | 0,715 | 0,710 | 0,922 |

- «Идеальная перестановка» — потолок: нужные тексты, найденные в топ-20, стоят на первых местах. Это не результат какой-то модели.
- Запас до потолка ≈ 0,2 (на HotpotQA ≈ 0,02). BERT на SciFact даже не лучше простого поиска (на тесте лучше на 26 вопросах, хуже на 37).
  Поэтому здесь у дорогой модели есть шанс заметно помочь. Хорошо ли она его использует, пока неизвестно: её на SciFact ещё не запускали.
- Размер запроса к 120B: в среднем 4097 токенов на вход (измерено на проверочных) плюс ответ около 200–300.
  Итого ≈ 4,4 тысячи токенов на утверждение, то есть ~45 утверждений в сутки на один ключ.

Если у тебя те же числа, идём дальше:

```bash
rerank --config config.scifact.toml estimate --answer-queries 0 --rerank-tokens 4400   # 4097 на вход измерено, ответ ~300 оценён
rerank --config config.scifact.toml run --split validation --method llm --allow-api   # 50 запросов, около 1,1 суток лимита
rerank --config config.scifact.toml tune
rerank --config config.scifact.toml run --split test --method llm --allow-api         # 150 запросов, около 3,3 суток лимита на один ключ; можно разделить между двумя ключами, как в README (`--part i/n`, `merge`)
rerank --config config.scifact.toml run --split test --method all --allow-api         # каскады из кэша, лимит не тратят
rerank --config config.scifact.toml evaluate --split test
rerank --config config.scifact.toml controls
rerank --config config.scifact.toml report
```

Для долгих запусков те же циклы, что в README: `caffeinate -i bash -c 'until rerank --config config.scifact.toml run ...; do sleep 600; done'`.
