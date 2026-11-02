"""Merge all raters into one table and compute inter-rater agreement.

  python src/agreement.py --candidates data/candidate_pairs.csv --rater1 data/annotations.csv \
      --others data/rater2.csv data/rater3.csv --blind_from 30

`--rater1` is your own annotations.csv (needs sentence1, sentence2, my_score).
Each file in `--others` needs columns pair_id and score (whole numbers 0-4).
Outputs: data/ratings_wide.csv, results/tables/agreement.csv, agreement_summary.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def pairwise_stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.all(a == np.round(a)) and np.all(b == np.round(b))):
        raise ValueError("ratings must be whole numbers 0-4")
    return {
        "n": len(a),
        "spearman": float(spearmanr(a, b)[0]),
        "qwk": float(cohen_kappa_score(a.astype(int), b.astype(int),
                                       labels=[0, 1, 2, 3, 4], weights="quadratic")),
        "exact": float(np.mean(a == b)),
        "within1": float(np.mean(np.abs(a - b) <= 1)),
        "mad": float(np.mean(np.abs(a - b))),
    }


def build_wide(candidates, rater1, others=(), blind_from=None, score_col="my_score"):
    cand = pd.read_csv(candidates).reset_index().rename(columns={"index": "pair_id"})
    ann = pd.read_csv(rater1)
    m = ann.merge(cand[["pair_id", "sentence1", "sentence2"]],
                  on=["sentence1", "sentence2"], how="left")
    if m["pair_id"].isna().any():
        raise ValueError("some annotated pairs were not found in the candidates file")
    if m["pair_id"].duplicated().any():
        raise ValueError("duplicate pair ids after merging")
    wide = cand.merge(m[["pair_id", score_col]].rename(columns={score_col: "rater1"}),
                      on="pair_id", how="inner")
    names = ["rater1"]
    for path in others:
        name = os.path.splitext(os.path.basename(path))[0]
        r = pd.read_csv(path)[["pair_id", "score"]].rename(columns={"score": name})
        r[name] = pd.to_numeric(r[name], errors="coerce")
        wide = wide.merge(r.dropna(), on="pair_id", how="left")
        names.append(name)
    wide["gold"] = wide[names].mean(axis=1)
    if blind_from is not None:
        wide["blind"] = wide["pair_id"] >= blind_from
    return wide.sort_values("pair_id").reset_index(drop=True), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidate_pairs.csv")
    ap.add_argument("--rater1", default="data/annotations.csv")
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--blind_from", type=int, default=None,
                    help="pair_id from which the expected band was hidden while rating")
    ap.add_argument("--out", default="data/ratings_wide.csv")
    a = ap.parse_args()

    wide, names = build_wide(a.candidates, a.rater1, a.others, a.blind_from)
    wide.to_csv(a.out, index=False)
    print(f"{len(wide)} pairs, raters: {names} -> {a.out}")
    os.makedirs("results/tables", exist_ok=True)

    rows = []
    for x, y in itertools.combinations(names, 2):
        both = wide[[x, y]].dropna()
        if len(both) >= 5:
            rows.append({"rater_a": x, "rater_b": y, **pairwise_stats(both[x], both[y])})
    pd.DataFrame(rows).to_csv("results/tables/agreement.csv", index=False)
    summary = {}
    if rows:
        df = pd.DataFrame(rows)
        print(df.round(3).to_string(index=False))
        summary = {"mean_pairwise_spearman": df.spearman.mean(), "mean_pairwise_qwk": df.qwk.mean()}
    if len(names) >= 2:
        try:
            import krippendorff
            data = wide[names].to_numpy(float).T
            summary["krippendorff_alpha_ordinal"] = float(
                krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"))
        except ImportError:
            print("pip install krippendorff to get alpha")
    pd.DataFrame([summary]).to_csv("results/tables/agreement_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
