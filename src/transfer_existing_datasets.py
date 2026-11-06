"""Apply a released SimCSE checkpoint to five existing datasets it was never evaluated on.

  python src/transfer_existing_datasets.py                 # all five
  python src/transfer_existing_datasets.py --only quora biosses

Pair construction
  quora       2,000 random question pairs; label = is_duplicate        (point-biserial)
  biosses     all 100 pairs; label = expert score 0-4                   (Spearman)
  codesearch  300 Python functions; (code, own docstring)=1 vs (code, another docstring)=0
  amazon      1,000 reviews; (title, own text)=1 vs (title, another review's text)=0
  cuad        clause pairs from the same legal category =1, different category =0
Mismatched pairs use a cyclic shift, so no item is ever paired with itself.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr

from stats_utils import bootstrap_ci

SEED = 42


def shift(items, rng):
    """Cyclic shift by a random non-zero offset (no fixed points)."""
    k = int(rng.integers(1, len(items)))
    return items[k:] + items[:k]


def matched_mismatched(a, b, rng):
    return pd.DataFrame({"sentence1": list(a) + list(a),
                         "sentence2": list(b) + shift(list(b), rng),
                         "label": [1] * len(a) + [0] * len(a)})


def load_quora(n, rng):
    from datasets import load_dataset
    ds = load_dataset("AlekseyKorshuk/quora-question-pairs", split="train")
    s = ds.select(rng.permutation(len(ds))[:n].tolist())
    return pd.DataFrame({"sentence1": s["question1"], "sentence2": s["question2"],
                         "label": s["is_duplicate"]}), "pointbiserial"


def load_biosses(n, rng):
    from datasets import load_dataset
    b = load_dataset("tabilab/biosses", split="train")
    return pd.DataFrame({"sentence1": b["sentence1"], "sentence2": b["sentence2"],
                         "label": b["score"]}), "spearman"


def load_codesearch(n, rng):
    from datasets import load_dataset
    ds = load_dataset("code-search-net/code_search_net", "python", split="train[:2000]")
    s = ds.select(rng.permutation(len(ds))[:n].tolist())
    keep = [(c, d) for c, d in zip(s["func_code_string"], s["func_documentation_string"]) if d.strip()]
    return matched_mismatched([c for c, _ in keep], [d for _, d in keep], rng), "pointbiserial"


def load_amazon(n, rng):
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/amazon_polarity", split="train[:5000]")
    s = ds.select(rng.permutation(len(ds))[:n].tolist())
    return matched_mismatched(s["title"], s["content"], rng), "pointbiserial"


def load_cuad(n, rng):
    from datasets import load_dataset
    ds = load_dataset("theatticusproject/cuad-qa", split="train", revision="refs/convert/parquet")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
    df = pd.DataFrame({"category": ds["question"], "clause": [a["text"][0] for a in ds["answers"]]})
    counts = df["category"].value_counts()
    good = counts[counts >= 20].index
    df = df[df["category"].isin(good)].reset_index(drop=True)
    rows = []
    for cat in good:
        cl = df.loc[df["category"] == cat, "clause"].tolist()
        for _ in range(20):
            i, j = rng.choice(len(cl), 2, replace=False)
            rows.append((cl[i], cl[j], 1))
    for _ in range(len(rows)):
        while True:
            i, j = rng.choice(len(df), 2, replace=False)
            if df.loc[i, "category"] != df.loc[j, "category"]:
                break
        rows.append((df.loc[i, "clause"], df.loc[j, "clause"], 0))
    return pd.DataFrame(rows, columns=["sentence1", "sentence2", "label"]), "pointbiserial"


LOADERS = {"quora": (load_quora, 2000), "biosses": (load_biosses, 100),
           "codesearch": (load_codesearch, 300), "amazon": (load_amazon, 1000),
           "cuad": (load_cuad, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(LOADERS))
    ap.add_argument("--model", default="princeton-nlp/sup-simcse-bert-base-uncased")
    ap.add_argument("--outdir", default="results/existing")
    a = ap.parse_args()
    from simcse_utils import default_pooler, load_encoder, pair_cosine
    tok, model, device = load_encoder(a.model)
    pooler = default_pooler(a.model)
    os.makedirs(a.outdir, exist_ok=True)
    summary = []
    for name in a.only:
        loader, n = LOADERS[name]
        df, kind = loader(n, np.random.default_rng(SEED))
        df["simcse_sim"] = pair_cosine(df["sentence1"].tolist(), df["sentence2"].tolist(),
                                       tok, model, device, pooler)
        df.to_csv(f"{a.outdir}/{name}_pairs_scored.csv", index=False)
        est, lo, hi = bootstrap_ci(df["simcse_sim"], df["label"], kind)
        row = {"dataset": name, "n_pairs": len(df), "metric": kind,
               "estimate": round(est, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}
        if kind == "pointbiserial":
            row["mean_label1"] = round(df.loc[df.label == 1, "simcse_sim"].mean(), 3)
            row["mean_label0"] = round(df.loc[df.label == 0, "simcse_sim"].mean(), 3)
        summary.append(row)
        print(row)
    pd.DataFrame(summary).to_csv(f"{a.outdir}/summary.csv", index=False)


if __name__ == "__main__":
    main()
