"""Create a blind rating sheet for additional human raters (open it in Google Sheets).

The sheet contains NO expected band, NO model scores and NO other rater's scores,
and the rows are shuffled. Raters fill the empty `score` column with whole numbers 0-4.

  python src/export_annotation_sheet.py --out data/sheet_rater2.csv
  python src/export_annotation_sheet.py --out data/sheet_rater2.csv --n 30   # stratified subset (about 30 rows)
"""
import argparse

import pandas as pd


def make_sheet(candidates, n=None, seed=7, from_id=0):
    cand = pd.read_csv(candidates).reset_index().rename(columns={"index": "pair_id"})
    cand = cand[cand["pair_id"] >= from_id]  # e.g. --from_id 60 exports only the new pairs
    if n and n < len(cand):
        parts = [g.sample(n=max(1, round(n * len(g) / len(cand))), random_state=seed)
                 for _, g in cand.groupby("expected_band")]
        cand = pd.concat(parts)
    sheet = cand[["pair_id", "sentence1", "sentence2"]].sample(frac=1, random_state=seed)
    sheet["score"] = ""
    return sheet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidate_pairs.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from_id", type=int, default=0, help="only export pairs with pair_id >= this")
    a = ap.parse_args()
    s = make_sheet(a.candidates, a.n or None, a.seed, a.from_id)
    s.to_csv(a.out, index=False)
    print(f"wrote {len(s)} rows to {a.out}")
