"""Merge the 40 new ratings (pair_id 60-99) into the existing rating files.

Rater 1 (you):   python src/append_ratings.py --mode rater1 --old data/annotations.csv \
                     --new data/sheet_me_new.csv --out data/annotations.csv
Raters 2 and 3:  python src/append_ratings.py --mode other --old data/rater2.csv \
                     --new data/rater2_new.csv --out data/rater2.csv
`--new` is the returned sheet (columns pair_id, score). Every id 60-99 must be rated.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rater1", "other"], required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default="data/candidate_pairs.csv")
    ap.add_argument("--first_new_id", type=int, default=60)
    a = ap.parse_args()
    new = pd.read_csv(a.new)[["pair_id", "score"]]
    new["score"] = pd.to_numeric(new["score"], errors="coerce")
    new = new[new["pair_id"] >= a.first_new_id]
    cand = pd.read_csv(a.candidates)
    need = set(range(a.first_new_id, len(cand)))
    miss = sorted(need - set(new.dropna()["pair_id"]))
    if miss:
        raise SystemExit(f"missing/blank scores for pair_id: {miss}")
    new = new.dropna()
    if not new["score"].isin([0, 1, 2, 3, 4]).all():
        raise SystemExit("scores must be whole numbers 0-4")
    old = pd.read_csv(a.old)
    if a.mode == "other":
        keep = old[old["pair_id"] < a.first_new_id][["pair_id", "score"]]
        out = pd.concat([keep, new], ignore_index=True)
    else:
        add = cand.reset_index().rename(columns={"index": "pair_id"}).merge(new, on="pair_id")
        add = add.rename(columns={"score": "my_score"})[["sentence1", "sentence2", "expected_band", "my_score"]]
        base = old[["sentence1", "sentence2", "expected_band", "my_score"]]
        out = pd.concat([base, add], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"{a.out}: {len(old)} -> {len(out)} rows")


if __name__ == "__main__":
    main()
