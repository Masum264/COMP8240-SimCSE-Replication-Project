"""Extra checks on the main results (paired tests, judge as a fourth rater, blind-only gold).

  python src/extra_analysis.py            # prints and writes results/tables/extra_*.csv

1. Paired bootstrap of rho(LLM, gold) - rho(SimCSE, gold) on each evaluation set. The two
   correlations share the same pairs, so the difference must be bootstrapped jointly, not read
   off two overlapping intervals.
2. Judge order consistency: Spearman between the judge's score for (A,B) and for (B,A).
3. Own data only: every rater, the judge and SimCSE scored against the mean of the OTHER
   human raters ("leave-one-out"), so the judge is compared with a human on equal terms.
4. Own data only: SimCSE and the judge against a gold built from the two raters who were
   always blind (rater2, rater3), as a check on the non-blind ratings of rater1.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import load_set

N_BOOT, SEED = 5000, 42


def rho(a, b):
    return spearmanr(a, b)[0]


def paired_diff(df, x="llm_score", y="simcse_sim", g="gold"):
    d = df[[x, y, g]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(d), len(d))
        s = d.iloc[i]
        diffs.append(rho(s[x], s[g]) - rho(s[y], s[g]))
    diffs = np.array([v for v in diffs if v == v])
    est = rho(d[x], d[g]) - rho(d[y], d[g])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return est, lo, hi, float((diffs <= 0).mean())


def main():
    os.makedirs("results/tables", exist_ok=True)
    rows = []
    for name in ["stsb_sample", "biosses", "own"]:
        if not os.path.exists(f"data/eval/{name}.csv"):
            continue
        df = load_set(name)
        est, lo, hi, p = paired_diff(df)
        row = {"set": name, "n": len(df), "diff_llm_minus_simcse": round(est, 3),
               "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "boot_share_diff_le_0": round(p, 4)}
        lp = f"results/llm/{name}.csv"
        if os.path.exists(lp):
            l = pd.read_csv(lp)
            if "llm_ba" in l and l["llm_ba"].notna().any():
                row["judge_order_spearman"] = round(rho(l["llm_ab"], l["llm_ba"]), 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv("results/tables/extra_paired.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    wide = pd.read_csv("data/ratings_wide.csv")
    own = load_set("own").merge(wide[["pair_id", "rater1", "rater2", "rater3"]], on="pair_id")
    raters = ["rater1", "rater2", "rater3"]
    loo = []
    for r in raters:
        others = own[[o for o in raters if o != r]].mean(axis=1)
        loo.append({"scorer": r, "vs_mean_of_other_raters": round(rho(own[r], others), 3)})
    # model scores vs the same leave-one-out golds, averaged over the three choices
    for col, label in [("llm_score", "LLM judge"), ("simcse_sim", "SimCSE")]:
        vals = [rho(own[col], own[[o for o in raters if o != r]].mean(axis=1)) for r in raters]
        loo.append({"scorer": label, "vs_mean_of_other_raters": round(float(np.mean(vals)), 3)})
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv("results/tables/extra_leave_one_out.csv", index=False)
    print("\nOwn data, each scorer vs mean of the other two human raters (model rows: mean over 3 choices)")
    print(loo_df.to_string(index=False))

    blind_gold = own[["rater2", "rater3"]].mean(axis=1)
    bl = pd.DataFrame([{"scorer": "LLM judge", "spearman_vs_rater2+3": round(rho(own["llm_score"], blind_gold), 3)},
                       {"scorer": "SimCSE", "spearman_vs_rater2+3": round(rho(own["simcse_sim"], blind_gold), 3)}])
    bl.to_csv("results/tables/extra_blind_gold.csv", index=False)
    print("\nOwn data vs gold from the two always-blind raters")
    print(bl.to_string(index=False))


if __name__ == "__main__":
    main()
