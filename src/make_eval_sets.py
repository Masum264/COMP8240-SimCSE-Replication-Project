"""Create the three evaluation inputs used for SimCSE scores and the LLM judge.

  data/eval/stsb_sample.csv   300 random STS-B test pairs (the paper's own data)
  data/eval/biosses.csv       all 100 BIOSSES pairs (expert 0-4 gold)
  data/eval/own.csv           your pairs, gold = mean human rating (needs data/ratings_wide.csv)

Columns: pair_id, sentence1, sentence2, gold
"""
import argparse
import os

import pandas as pd

STSB_DEFAULT = "third_party/SimCSE/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv"


def load_stsb(path):
    """STS-B file: genre, file, year, id, score, sentence1, sentence2 (tab separated)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7:
                try:
                    rows.append((float(parts[4]), parts[5], parts[6]))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["gold", "sentence1", "sentence2"])
    return df.reset_index().rename(columns={"index": "pair_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stsb", default=STSB_DEFAULT)
    ap.add_argument("--stsb_n", type=int, default=300)
    ap.add_argument("--ratings", default="data/ratings_wide.csv")
    ap.add_argument("--outdir", default="data/eval")
    ap.add_argument("--skip_biosses", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cols = ["pair_id", "sentence1", "sentence2", "gold"]

    if os.path.exists(a.stsb):
        df = load_stsb(a.stsb)
        df = df.sample(n=min(a.stsb_n, len(df)), random_state=42).sort_values("pair_id")
        df[cols].to_csv(f"{a.outdir}/stsb_sample.csv", index=False)
        print("stsb_sample:", len(df))
    else:
        print("STS-B file not found (run src/replicate_sts.sh first):", a.stsb)

    if not a.skip_biosses:
        from datasets import load_dataset
        b = load_dataset("tabilab/biosses", split="train").to_pandas()
        out = pd.DataFrame({"pair_id": range(len(b)), "sentence1": b["sentence1"],
                            "sentence2": b["sentence2"], "gold": b["score"]})
        out.to_csv(f"{a.outdir}/biosses.csv", index=False)
        print("biosses:", len(out))

    if os.path.exists(a.ratings):
        w = pd.read_csv(a.ratings)
        w[cols].to_csv(f"{a.outdir}/own.csv", index=False)
        print("own:", len(w))
    else:
        print("ratings file not found (run src/agreement.py first):", a.ratings)


if __name__ == "__main__":
    main()
