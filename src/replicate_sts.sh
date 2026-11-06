#!/usr/bin/env bash
# Re-run the authors' STS evaluation (Gao et al., 2021) on four released checkpoints.
# Run from the repository root (Linux/Colab). Needs a GPU for reasonable speed.
set -euo pipefail
ROOT="$(pwd)"
TP="${THIRD_PARTY:-third_party}"   # set THIRD_PARTY=/content/drive/... to keep the clone on Drive
mkdir -p "$TP" results/replication
[ -d "$TP/SimCSE" ] || git clone https://github.com/princeton-nlp/SimCSE.git "$TP/SimCSE"
cd "$TP/SimCSE"

# The repo's requirements.txt pins 2021 versions that no longer build; use current ones.
pip install -q "transformers>=4.30" --upgrade
pip install -q scikit-learn scipy prettytable

# Compatibility patch: modern transformers removed tokenizer.batch_encode_plus.
# This is the ONLY change made to the authors' code.
sed -i 's/tokenizer.batch_encode_plus(/tokenizer(/' evaluation.py

(cd SentEval/data/downstream && bash download_dataset.sh)

run() {  # $1 = short name, $2 = HF model, $3 = pooler
  python evaluation.py --model_name_or_path "$2" --pooler "$3" \
      --task_set sts --mode test 2>&1 | tee "$ROOT/results/replication/$1.txt"
}
run sup-bert-base      princeton-nlp/sup-simcse-bert-base-uncased    cls
run unsup-bert-base    princeton-nlp/unsup-simcse-bert-base-uncased  cls_before_pooler
run sup-roberta-base   princeton-nlp/sup-simcse-roberta-base         cls
run unsup-roberta-base princeton-nlp/unsup-simcse-roberta-base       cls_before_pooler
