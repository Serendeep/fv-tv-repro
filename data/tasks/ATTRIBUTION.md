# Task data provenance

All `*.json` files in this directory are copied verbatim (no reformatting) from:

- Repo: https://github.com/ericwtodd/function_vectors (Todd et al., "Function Vectors in Large
  Language Models," ICLR 2024)
- Paths: `dataset_files/abstractive/*.json` and `dataset_files/extractive/*.json`
- License: MIT (see upstream `LICENSE`; copyright (c) 2023 Eric Todd)
- Recorded source snapshot: shallow clone of `main` taken 2026-07-12.
  The immutable upstream commit hash was not recorded.

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

The selected files contain 197–5,199 pairs each. Evaluation splits contain 15 or 25
items depending on the arm (see `RUNS.md`), with evaluation rows excluded from
demonstrations. Answers are not filtered by word or token length; first-token
accuracy does not establish complete-answer accuracy.

## Upstream dataset credit

The immediate source is Todd et al., *Function Vectors in Large Language Models*,
ICLR 2024. [Appendix E](https://arxiv.org/abs/2310.15213) records the underlying
dataset sources, including Nguyen et al. for antonym/synonym, Conneau et al. for
translation, and Hernandez et al. for several knowledge tasks. These data were
not newly collected for this reproduction. See that appendix for source-specific
construction details and references.
