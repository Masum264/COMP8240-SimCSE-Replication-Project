#!/usr/bin/env bash
# Full pipeline, in order. Run from the repository root on a GPU machine (e.g. Colab).
# Steps that need other people (extra raters) are marked.
set -euo pipefail
export PYTHONPATH=src

# 1. Replicate the paper's STS results with four released checkpoints
bash src/replicate_sts.sh
python src/parse_replication.py

# 2. Apply SimCSE to five existing datasets
python src/transfer_existing_datasets.py

# 3. Own dataset: (optional) verify the pairs regenerate, export the blind rating sheet
# (build_dataset.py regenerates the 60 batch-1 pairs, so compare with the first 60 rows of the final file)
python -c "import pandas as pd; pd.read_csv('data/candidate_pairs.csv').head(60).to_csv('/tmp/first60.csv', index=False)"
python src/build_dataset.py --from_json data/raw_texts.json --verify /tmp/first60.csv
python src/export_annotation_sheet.py --out data/sheet_for_raters.csv
#    Extension 60 -> 100 pairs (first 60 untouched):
#      python src/extend_dataset.py --raw data/raw_texts.json --candidates data/candidate_pairs.csv
#      python src/export_annotation_sheet.py --from_id 60 --out data/sheet_new40.csv   # one copy per rater (incl. you)
#      python src/append_ratings.py --mode rater1 --old data/annotations.csv --new data/me_new.csv --out data/annotations.csv
#      python src/append_ratings.py --mode other  --old data/rater2.csv --new data/rater2_new.csv --out data/rater2.csv
#    >>> give data/sheet_for_raters.csv to each extra rater; save their returned
#    >>> files as data/rater2.csv, data/rater3.csv (columns: pair_id, score)

# 4. Agreement + evaluation sets
python src/agreement.py --others data/rater2.csv data/rater3.csv --blind_from 30   # three raters, as in the report
python src/make_eval_sets.py

# 5. SimCSE scores and LLM-judge scores for the three evaluation sets
for s in stsb_sample biosses own; do
  python src/simcse_scores.py --input data/eval/$s.csv --out results/simcse/$s.csv
  python src/llm_judge.py --input data/eval/$s.csv --out results/llm/$s.csv --load_4bit
done

# 6. Final tables
python src/analysis.py

# 7. Extra analyses and the tables/figures used in the report
python src/extra_analysis.py        # paired bootstrap, judge consistency, leave-one-out
python src/clean_subset.py          # personal-information screen and clean-subset comparison
python src/make_figures.py          # results/figures/scatter_own.pdf
python src/make_snippet_table.py    # results/tables/snippets_own.tex
python src/make_existing_snippets.py
python src/make_original_snippets.py --root "${THIRD_PARTY:-third_party}/SimCSE/SentEval/data/downstream"
