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
| `src/build_dataset.py`, `src/new_sources.py`, `src/extend_dataset.py` | how the 100 distinct candidate pairs were built from thirteen MIT OCW syllabi (batch 1: five syllabi, batch 2: eight more) |
| `src/append_ratings.py` | merges the second batch of ratings into the rating files |
| `src/export_annotation_sheet.py`, `src/agreement.py` | blind rating sheet for extra raters; agreement statistics |
| `src/extra_analysis.py`, `src/clean_subset.py` | paired bootstrap, leave-one-out rater comparison, personal-information screen (`data/row_flags.csv`) |
| `src/make_snippet_table.py`, `src/make_original_snippets.py`, `src/make_existing_snippets.py`, `src/make_figures.py` | LaTeX snippet tables and the scatter figure used in the report |
| `src/make_eval_sets.py`, `src/simcse_scores.py`, `src/llm_judge.py`, `src/analysis.py` | evaluation sets, scores, final tables |
| `data/` | `raw_texts.json`, `candidate_pairs.csv`, `annotations.csv`, rater files, `eval/` (see `data/README.md`) |
| `docs/` | annotation guidelines, information sheet for raters |
| `results/` | all generated outputs and tables used in the report |

## Environment
Developed on Google Colab with an NVIDIA T4 GPU: Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0 build),
`transformers` 5.22.1, `scikit-learn` 1.6.1, `scipy` 1.16.3.
```
pip install -r requirements.txt
```
The authors' `requirements.txt` pins 2021 versions (e.g. `tokenizers==0.9.4`) that no longer build on
current Python. `src/replicate_sts.sh` therefore uses current packages and makes **one** code change:
`tokenizer.batch_encode_plus(` -> `tokenizer(` in `evaluation.py` (the method was removed).

## Reproduce
Tested on Google Colab with a T4 GPU.

1. Open a Colab notebook and choose Runtime > Change runtime type > T4 GPU.
2. Clone the repo and install the packages:
```
!git clone https://github.com/Masum264/COMP8240-SimCSE-Replication-Project
%cd COMP8240-SimCSE-Replication-Project
!pip install -q -r requirements.txt bitsandbytes
```
3. Quick check, no GPU needed (a few minutes). Expected: mean Spearman 0.729, QWK 0.690, alpha 0.652.
```
%env PYTHONPATH=src
!python src/agreement.py --others data/rater2.csv data/rater3.csv --blind_from 30
!python src/analysis.py
```
4. Full pipeline (replication, five existing datasets, SimCSE and judge scores, tables):
```
!bash run_all.sh
```

All sampling uses fixed seeds. LLM-judge outputs are deterministic for the local backend (`results/llm/*.csv.meta.json` records model, prompt hash and date). Pipeline test without a GPU: `python src/llm_judge.py --backend dummy ...`

## Data and licences
See `data/README.md` (datasheet). The new dataset's sentences come from MIT OpenCourseWare syllabus pages, which
state CC BY-NC-SA 4.0. The text is released with attribution to MIT OpenCourseWare, under the same licence, for
non-commercial use only. `data/row_flags.csv` marks 29 rows with likely person names or reference lines. To request
removal of a row, open an issue on this repository. The existing datasets (Quora, BIOSSES, CodeSearchNet, Amazon, CUAD,
and the STS-B sample) are not copied here: the scripts download them from their public sources.

## Citation
Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021.
