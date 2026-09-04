# fv-tv-repro

Code, data, and results for "Do Function Vectors and Task Vectors Generalize? A Reproducibility
Study Across Model Families and Scale" (BlackboxNLP 2026, Reproducibility Track). The paper
source and compiled PDF are in `paper/`.

Reproduction of function vectors (Todd et al., "Function Vectors in Large Language Models,"
ICLR 2024) and task vectors (Hendel et al., "In-Context Learning Creates Task Vectors," EMNLP
2023 Findings) across GPT-J-6B, Llama-3.1-8B, Gemma-2-9b-it, and Llama-3.1-70B, run via
NNsight and NDIF.

## Setup

```
uv venv
uv pip install nnsight torch numpy scipy pandas matplotlib
```

Remote runs (`--remote`) call NDIF-hosted models and need an NDIF API key:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("<your key>")
```

Gated Llama checkpoints need a Hugging Face token with access, read from the `HF_TOKEN`
environment variable:

```
export HF_TOKEN=<your token>
```

## Repo layout

- `src/fvtv/` - library code: task loading (`tasks.py`), function vectors (`fv.py`), task
  vectors (`tv.py`), controls (`controls.py`), ICL eval utilities (`eval_icl.py`), statistics
  (`stats.py`).
- `scripts/` - entry points: `run_grid.py` (main experiment grid), `analyze.py` /
  `extra_stats.py` / `revision_stats.py` (aggregate results into tables), `make_figure.py`
  (layer-profile figure), `make_extra_figures.py` (effect-summary and control figures),
  `gemma_diagnostics.py` / `gemma_alpha.py` / `gemma_crosstask_fv.py` (Gemma-2 FV-null
  diagnostics, the last one applying Todd et al.'s cross-task head selection at k in
  {10, 20, 40}), `tv_controls_extra.py` (cross-task, template-swap, and additive task-vector
  controls) with `tvextra_stats.py` and `per_task_tv_table.py` for their summaries.
- `data/tasks/` - task word-pair JSON files, provenance in `data/tasks/ATTRIBUTION.md`.
- `results/` - committed experiment outputs (`grid_*.json`, diagnostics JSON, aggregated CSVs).

## How to reproduce the tables

The paper's numbers come from the committed `results/` files; no GPU or NDIF access is needed
to regenerate them.

```
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/extra_stats.py
.venv/bin/python scripts/revision_stats.py
.venv/bin/python scripts/make_figure.py
.venv/bin/python scripts/make_extra_figures.py
.venv/bin/python scripts/panel_revision_stats.py
.venv/bin/python scripts/tvextra_stats.py
.venv/bin/python scripts/per_task_tv_table.py
```

`analyze.py` and `extra_stats.py` write `results/summary.csv` and `results/layer_profiles.csv`.
`revision_stats.py` prints the clustered-CI, CV co-primary, and protocol/coverage tables used
in the revision pass. `make_figure.py` renders the layer-profile figure from
`results/layer_profiles.csv`.

## How to re-run experiments

The commands above only re-derive tables from committed results. To regenerate the results
themselves, see `RUNS.md` for the exact worker invocations, including model, task, seed, and
sample-size arguments per shard.

## Claim map

- C1.1 (function vectors): `scripts/run_grid.py`, FV path implemented in `src/fvtv/fv.py`.
- C2.1 / C2.2 (task vectors): `src/fvtv/tv.py`.
- C3.1: `scripts/extra_stats.py`.
- Gemma-2 diagnostics: `scripts/gemma_diagnostics.py`, cross-task head selection in
  `scripts/gemma_crosstask_fv.py` (maps and head sets in `results/gemma_xtask_*.pt`).
- Task-vector controls beyond label shuffling: `scripts/tv_controls_extra.py`
  (`results/grid_*_tvextra.json`).

## Provenance note

Result rows record the git SHA at run time. Rows produced before a history cleanup carry SHAs
that no longer resolve in this repository; the code that produced every committed result is the
code at the initial release commit or earlier states of the same files. The analysis scripts
regenerate every number in the paper from the committed results.

## Citation

```bibtex
@inproceedings{rudraraju2026fvtv,
  title     = {Do Function Vectors and Task Vectors Generalize? A Reproducibility Study Across Model Families and Scale},
  author    = {Rudraraju, Serendeep},
  booktitle = {Proceedings of the 9th BlackboxNLP Workshop},
  year      = {2026}
}
```
