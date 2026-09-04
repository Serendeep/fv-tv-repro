# Runs

Exact worker invocations used to produce the committed `results/grid_*.json` and
`results/gemma_*` files. Run from the repo root with `uv run` (see README setup).
`--remote` requires an NDIF API key; gated Llama models also require `HF_TOKEN`.

```
# GPT-J (replication arm; 15-item splits, stride 4 after deployment instability)
uv run python scripts/run_grid.py --model gpt-j-6b --remote --tasks antonym,country-capital,english-french,present-past --seeds 0,1 --n-eval 15 --n-aie-trials 5 --sweep-stride 4 --aie-max-rows 16 --out results/grid_gpt-j-6b_shard0.json
uv run python scripts/run_grid.py --model gpt-j-6b --remote --tasks person-occupation,singular-plural,synonym,next_item --seeds 0,1 --n-eval 15 --n-aie-trials 5 --sweep-stride 4 --aie-max-rows 16 --out results/grid_gpt-j-6b_shard1.json
uv run python scripts/run_grid.py --model gpt-j-6b --remote --tasks capitalize,country-currency,english-german,prev_item,capitalize_first_letter,lowercase_first_letter,park-country,english-spanish --seeds 0,1 --n-eval 15 --n-aie-trials 5 --sweep-stride 4 --aie-max-rows 16 --out results/grid_gpt-j-6b_shard2.json
# Gemma-2 (16-row AIE cap; 5 trials)
uv run python scripts/run_grid.py --model gemma-2-9b-it --remote --tasks antonym,country-capital,english-french,present-past,person-occupation,singular-plural --seeds 0,1,2 --n-eval 25 --n-aie-trials 5 --sweep-stride 2 --aie-max-rows 16 --out results/grid_gemma-2-9b-it_shard0.json
uv run python scripts/run_grid.py --model gemma-2-9b-it --remote --tasks synonym,next_item,capitalize,country-currency,english-german,prev_item --seeds 0,1,2 --n-eval 25 --n-aie-trials 5 --sweep-stride 2 --aie-max-rows 16 --out results/grid_gemma-2-9b-it_shard1.json
# Llama-3.1-8B
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks antonym,country-capital,english-french,present-past --seeds 0,1,2 --n-eval 25 --n-aie-trials 10 --sweep-stride 2 --aie-max-rows 32 --out results/grid_llama-3.1-8b_shard0.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks person-occupation,singular-plural,synonym,next_item --seeds 0,1,2 --n-eval 25 --n-aie-trials 10 --sweep-stride 2 --aie-max-rows 32 --out results/grid_llama-3.1-8b_shard1.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks country-capital --seeds 0 --n-eval 25 --skip-fv --sweep-stride 2 --out results/grid_llama-3.1-8b_tv_probe_country-capital_s0.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks country-capital,english-french,present-past,next_item,synonym --seeds 0,1,2 --n-eval 25 --skip-fv --sweep-stride 2 --out results/grid_llama-3.1-8b_tv_newdata.json
# Reduced Llama-3.1-8B FV follow-up probes (not merged into the main grid)
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks country-capital --seeds 1 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_country-capital_s1.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks country-capital --seeds 2 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_country-capital_s2.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks english-french --seeds 0 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_english-french_s0.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks next_item --seeds 0 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_next_item_s0.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks next_item --seeds 1 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_next_item_s1.json
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks next_item --seeds 2 --n-eval 15 --n-ex 16 --n-aie-trials 3 --aie-layer-start 8 --aie-layer-end 25 --aie-max-rows 16 --fv-layers 12,16,20 --sweep-stride 4 --out results/grid_llama-3.1-8b_fv_probe_next_item_s2.json
# Llama-3.1-70B (TV only)
uv run python scripts/run_grid.py --model llama-3.1-70b --remote --tasks antonym,capitalize,capitalize_first_letter,country-capital,country-currency,english-french,english-german,english-spanish,lowercase_first_letter,next_item,park-country,person-occupation,present-past,prev_item,singular-plural,synonym --seeds 0,1,2 --n-eval 25 --skip-fv --sweep-stride 3 --out results/grid_llama-3.1-70b_shard0.json
# Gemma diagnostics
uv run python scripts/gemma_diagnostics.py antonym
uv run python scripts/gemma_diagnostics.py country-capital
uv run python scripts/gemma_alpha.py

# --- Camera-ready additions (Sept 2026, in response to reviews) ---
# Llama-3.1-8B FV: full protocol on five more tasks (moves C1.1 from insufficient data to pass)
uv run python scripts/run_grid.py --model llama-3.1-8b --remote --tasks country-capital,english-french,next_item,present-past,synonym --seeds 0,1,2 --n-eval 25 --n-aie-trials 10 --sweep-stride 2 --aie-max-rows 32 --out results/grid_llama-3.1-8b_shard2.json
# Gemma-2 cross-task head selection (Todd protocol, k in {10,20,40})
uv run python scripts/gemma_crosstask_fv.py
# Task-vector controls: cross-task donor, arrow-template, additive patching
uv run python scripts/tv_controls_extra.py --model gpt-j-6b --remote
uv run python scripts/tv_controls_extra.py --model llama-3.1-8b --remote
uv run python scripts/tv_controls_extra.py --model llama-3.1-70b --remote
# Summaries
uv run python scripts/tvextra_stats.py
uv run python scripts/per_task_tv_table.py
```

Coverage per arm is smaller than the task lists above where deployment outages ended a run
early. Committed result rows carry per-row `n_eval`, `seed`, and `git_sha`; read those fields
rather than assuming every shard completed its full task list.
