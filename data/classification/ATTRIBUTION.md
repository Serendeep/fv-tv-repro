# Sentiment data provenance

`sentiment.json` is the SST-2-derived sentiment file distributed with Todd et al.,
*Function Vectors in Large Language Models* (ICLR 2024), under
`dataset_files/abstractive/sentiment.json`. The retrieval URL, date, and SHA-256
appear in `sentiment-provenance.json`. We retain the downloaded bytes.

Credit the Stanford Sentiment Treebank (Socher et al., 2013) and Todd et al.'s
derived task collection when reusing these data. See Appendix E of Todd et al.
for the task's source attribution. The code license in this repository does not
replace the original dataset's terms.

This file contains 1,167 distinct inputs. The extension reserves 40 balanced
development inputs and 80 balanced test inputs, with the remainder available for
construction. Saved partitions record the exact assignment. Natural labels and
both arbitrary mappings use the same inputs. This directory is separate from
`data/tasks/` so the original word-pair task enumeration remains reproducible.
