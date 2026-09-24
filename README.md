# fv-tv-repro

Code, data, and results for "Do Function Vectors and Task Vectors Generalize? A Reproducibility
Study Across Model Families and Scale" (BlackboxNLP 2026, Reproducibility Track). The paper
source and compiled PDF are in `paper/`.

Reproduction of function vectors (Todd et al., "Function Vectors in Large Language Models,"
ICLR 2024) and task vectors (Hendel et al., "In-Context Learning Creates Task Vectors," EMNLP
2023 Findings) across GPT-J-6B, Llama-3.1-8B, Gemma-2-9b-it, and Llama-3.1-70B, run via
NNsight and NDIF. Matched follow-ups test base and instruction-tuned Gemma-2-9B
with native PyTorch hooks on Kaggle.

## Setup

```
uv sync --frozen
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
- `results/` - original experiment outputs, diagnostics, and derived CSVs.
  `heldout/` contains the 96-cell follow-up: predictions, partitions, selected
  layers, vector artifacts, and source snapshots.
- `results/extensions/` - 96 additional cells: 48 Gemma word-pair TVs,
  36 sentiment TVs across four checkpoints, and 12 Gemma FVs.
- `data/classification/` - sentiment inputs and attribution.
- `notebooks/` - frozen Kaggle execution notebooks for the Gemma follow-ups.
- `paper/` - manuscript source, compiled PDF, figures, and generated tables.
- `poster/` - A0 conference poster (`poster.html`, built `poster.pdf`); figures regenerate
  from `results/` with `make_poster_figures.py`.
- `tests/` - offline checks for analysis, split isolation, selection, and resuming runs.

## How to reproduce the tables

The paper's numbers come from the committed `results/` files; no GPU or NDIF access is needed
to regenerate them.

```
uv run python scripts/analyze.py
uv run python scripts/extra_stats.py
uv run python scripts/revision_stats.py
uv run python scripts/make_figure.py
uv run python scripts/make_extra_figures.py
uv run python scripts/panel_revision_stats.py
uv run python scripts/tvextra_stats.py
uv run python scripts/per_task_tv_table.py
uv run python scripts/paper_tables.py
uv run python scripts/analyze_heldout.py results/heldout
uv run python scripts/heldout_paper_tables.py --check
uv run python scripts/extension_paper_tables.py --check
uv run python scripts/make_sentiment_figure.py
uv run python scripts/verify_paper_results.py
uv run python scripts/paper_tables.py --check
uv run python -m unittest discover -s tests -v
```

`analyze.py` writes `results/summary.csv` and `results/layer_profiles.csv`.
`extra_stats.py` prints the layer-selection sensitivity and cross-method correlations.
`revision_stats.py` prints the clustered-CI, CV co-primary, and protocol/coverage tables used
in the revision pass. `make_figure.py` renders the layer-profile figure from
`results/layer_profiles.csv`. All main-grid summaries use the row selection in
`analyze.load_rows()`, which excludes diagnostics and reduced-protocol probes.
Task clusters and their values are sorted before bootstrapping so table intervals
are reproducible across input ordering.

`paper_tables.py` writes the LaTeX tables in `paper/generated/` from the saved
summaries and main-grid rows. `verify_paper_results.py` independently reads an
explicit set of main-grid JSON files and checks recovery calculations, clustered
intervals, effect sizes, and the displayed main and control tables. It needs no
model access. `paper_tables.py --check` detects stale generated table files.

Build the final PDF with a TeX installation that includes `latexmk` and BibTeX:

```
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

The metric is top-1 first-token accuracy, including for multi-token reference
answers. In the original exploratory grid, layer selection uses the evaluation
split. The held-out follow-up separates construction, development, and test inputs
and selects every method and control layer on development data. It covers eight
tasks and three construction seeds for FV and TV on GPT-J-6B and Llama-3.1-8B.

For the original grid, the seed-0 layer-selection analysis is a sensitivity check.
The two FV controls are averaged within each cell for paired differences and pooled
as separate observations for descriptive Cohen's d. Held-out comparisons average
seeds within tasks and give each task equal weight; confidence intervals bootstrap
tasks. `heldout_paper_tables.py --check` verifies all 96 cells and the three held-out
tables against saved predictions.

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

Original-grid result rows carry a `git_sha` field recording the commit reported by the
development checkout; this does not capture uncommitted changes. Those SHAs
come from the development tree and do not resolve in this repository, so treat the field as a
run identifier rather than something to check out.

Reproducing the paper does not depend on it. `scripts/analyze.py` and its companions regenerate
every table and figure from the committed `results/` files, and `RUNS.md` pins the model, tasks,
seeds, and sample sizes behind each one.

The held-out runs additionally save hashed source snapshots, exact partitions,
per-input token predictions, and frozen layer choices. The GPT-J FV arm used a
memory-recovery implementation preserved in its snapshot. Use the saved snapshots
when reproducing those exact runs; the current runner will reject a resume if its
source differs. New runs should use a fresh output directory.

The completed extensions are included in the paper.
`scripts/extension_paper_tables.py` verifies all 96 extension cells and regenerates
five tables from saved predictions. `scripts/make_sentiment_figure.py` generates
the main-text sentiment figure from the same verified results. Word-pair intervals resample tasks; sentiment
intervals resample inputs within sentiment classes, conditional on the fixed seeds
and mappings. See `results/extensions/README.md` for the archived artifacts
and `RUNS.md` for execution details.

## Citation

Accepted for presentation at BlackboxNLP 2026. This citation is provisional until
the proceedings record is available.


```bibtex
@misc{rudraraju2026fvtv,
  title     = {Do Function Vectors and Task Vectors Generalize? A Reproducibility Study Across Model Families and Scale},
  author    = {Rudraraju, Serendeep},
  note      = {Accepted for presentation at BlackboxNLP 2026},
  year      = {2026}
}
```
