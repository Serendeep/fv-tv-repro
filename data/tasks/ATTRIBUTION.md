# Task data provenance

All `*.json` files in this directory are copied verbatim (no reformatting) from:

- Repo: https://github.com/ericwtodd/function_vectors (Todd et al., "Function Vectors in Large
  Language Models," ICLR 2024)
- Paths: `dataset_files/abstractive/*.json` and `dataset_files/extractive/*.json`
- License: MIT (see upstream `LICENSE`; copyright (c) 2023 Eric Todd)
- Commit: shallow clone taken 2026-07-12, `main` branch HEAD at fetch time.

Format: each file is a flat JSON list of `{"input": ..., "output": ...}` word-pair objects.
This repo (`fv-tv-repro`) does not pre-split these into train/valid/test; `src/fvtv/tasks.py`
does seeded train-pool / eval-split sampling at run time.

## Selected 16 tasks (of Todd's ~57), by category

| Category | Tasks |
|---|---|
| Linguistic | antonym, synonym, present-past, singular-plural |
| Knowledge | country-capital, country-currency, person-occupation, park-country |
| Translation | english-french, english-spanish, english-german |
| Algorithmic | capitalize, capitalize_first_letter, lowercase_first_letter, next_item, prev_item |

Selection: tasks with >=200 pairs (enough for 10-shot demos plus evaluation splits of 15-25
items depending on arm (see RUNS.md), with no overlap) and short single/few-token answers, so
top-1 first-token accuracy stays meaningful.
Translation has only 3 tasks because Todd's repo only ships en-fr/en-es/en-de.
