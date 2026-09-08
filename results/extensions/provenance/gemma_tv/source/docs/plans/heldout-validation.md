# Held-out validation of function and task vectors

Recorded September 5, 2026, before outcome-bearing runs of this protocol. Camera-ready deadline: September 10. This is a prospective follow-up within datasets already explored in the paper, not an untouched external benchmark or an exact reproduction of the original papers.

## Question

Do the positive FV and TV effects survive separation of vector construction, layer selection and final evaluation, with controls given their own development-set layer selection?

## Fixed scope

GPT-J-6B and Llama-3.1-8B; eight shared tasks: antonym, country-capital, english-french, present-past, person-occupation, singular-plural, synonym and next_item. Three construction seeds: 100, 101, 102. All cells are retained regardless of baseline headroom or outcome. Missing cells and operational failures will be reported.

For each task, group rows by exact input string, then assign 50 distinct inputs to final test and 25 to development using split seed 20260905. The remaining inputs form the construction pool. Duplicate inputs never cross partitions; duplicate identical pairs are removed, while alternate answers remain in the same partition. The partition is fixed across models and construction seeds. Record exact row membership and data/code hashes. These tests reuse the task datasets from the original study, which limits independence from earlier research decisions.

TV extraction and the ICL baseline use the same ten demonstrations from construction only, with a separate construction-pool dummy query. FV means and AIE use construction only, including all internally sampled AIE queries. Retain existing per-task top-10 FV construction, 32 mean prompts and 10 AIE trials. This addresses split leakage, not every fidelity difference from Todd et al.

Sweep every fourth layer plus the final layer on development only, with ties resolved toward the lower layer. Select each method and each control independently. Freeze these choices before evaluating any final-test prompts. Report controls both at their own development-selected layer (primary) and at the corresponding real method's development-selected layer (secondary). No final-test layer sweep.

FV controls: norm-matched random vector and random ten-head vector. TV controls: three ordinary label permutations, keeping the same demonstrations and dummy query. Record permutations and surviving correct labels; do not select the strongest or weakest permutation after testing.

## Outcomes and analysis

Primary outcome: final-test first-token accuracy difference between each real intervention and its average independently selected controls. Report raw zero-shot and ICL accuracy, each method and control, per-task results, and absolute gains over zero shot. Average construction seeds within each task, then weight tasks equally. Bootstrap the eight task-level differences for descriptive 95% intervals, with a warning about the small number of task clusters. Seeds reuse the same test inputs and are not additional independent task samples. No pass/fail grading or outcome-dependent task filtering.

Recovery ratios are secondary, undefined when ICL minus zero-shot accuracy is nonpositive; report the denominator and count undefined cells. This follow-up still measures first-token accuracy, not full-answer correctness. No claim of a scale effect or an explanation of the Gemma null follows from these runs.

## Execution and verification

Validate split disjointness, selection independence, matching of demos and controls, resume integrity and per-item outputs with local tests. Run a technical smoke check on GPT-J; do not use its performance to revise task scope or selection rules. Run TV first, followed by FV, with incremental checkpoints. Confirm Llama availability before starting its arm. Persist source snapshots, arguments, split memberships and predictions. Independently recompute all reported accuracies and paired differences from saved token predictions before adding the completed follow-up to the paper.

If service availability prevents completion, publish the achieved coverage and errors. Do not silently replace tasks/models or fold partial results into the original main grid. Preserve the original results separately.
