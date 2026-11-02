# SimCSE reproduction and extension (COMP8240 final project)

Reproduces *SimCSE: Simple Contrastive Learning of Sentence Embeddings* (Gao, Yao & Chen,
EMNLP 2021, https://arxiv.org/abs/2104.08821), applies it to five existing datasets and to a
small new sentence-similarity dataset built from university course outlines, and compares
SimCSE with an LLM-as-judge.

## Layout
| Path | What it is |
|---|---|
| `src/replicate_sts.sh` | clones the authors' code, applies one compatibility patch, evaluates 4 released checkpoints |
| `src/parse_replication.py` | compares our numbers with Table 5 of the paper |
| `src/transfer_existing_datasets.py` | SimCSE on Quora, BIOSSES, CodeSearchNet, Amazon, CUAD |
| `src/build_dataset.py` | how the candidate pairs were built from five MIT OCW syllabi |
| `src/export_annotation_sheet.py`, `src/agreement.py` | blind rating sheet for extra raters; agreement statistics |
| `src/make_eval_sets.py`, `src/simcse_scores.py`, `src/llm_judge.py`, `src/analysis.py` | evaluation sets, scores, final tables |
| `data/` | `raw_texts.json`, `candidate_pairs.csv`, `annotations.csv`, rater files, `eval/` (see `data/README.md`) |
| `docs/` | annotation guidelines, information sheet for raters |
| `results/` | all generated outputs and tables used in the report |

## Environment
Developed on Google Colab (GPU runtime). Record your own versions after a successful run:
`pip freeze > environment_freeze.txt` and commit it.
```
pip install -r requirements.txt
```
The authors' `requirements.txt` pins 2021 versions (e.g. `tokenizers==0.9.4`) that no longer build on
current Python. `src/replicate_sts.sh` therefore uses current packages and makes **one** code change:
`tokenizer.batch_encode_plus(` -> `tokenizer(` in `evaluation.py` (the method was removed).

## Reproduce
Run `run_all.sh` from the repository root, or each step separately (see comments in that file).
All sampling uses fixed seeds. LLM-judge outputs are deterministic for the local backend
(`results/llm/*.csv.meta.json` records model, prompt hash and date).
Quick pipeline test without a GPU: `python src/llm_judge.py --backend dummy ...`.

## Data and licences
See `data/README.md`. Source texts keep their original licences; check each source before reuse.

## Citation
Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021.
