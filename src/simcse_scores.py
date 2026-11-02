"""Add SimCSE cosine similarity to an eval CSV.

  python src/simcse_scores.py --input data/eval/own.csv --out results/simcse/own.csv
"""
import argparse
import os

import pandas as pd

from simcse_utils import default_pooler, load_encoder, pair_cosine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="princeton-nlp/sup-simcse-bert-base-uncased")
    ap.add_argument("--pooler", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    pooler = a.pooler or default_pooler(a.model)
    tok, model, device = load_encoder(a.model)
    sims = pair_cosine(df["sentence1"].tolist(), df["sentence2"].tolist(), tok, model, device, pooler)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame({"pair_id": df["pair_id"], "simcse_sim": sims}).to_csv(a.out, index=False)
    print(f"wrote {len(df)} scores to {a.out} (model={a.model}, pooler={pooler})")
