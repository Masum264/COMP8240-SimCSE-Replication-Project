"""Final tables: SimCSE and LLM judge against gold, with bootstrap 95% CIs.

Needs, per set in {stsb_sample, biosses, own}:
  data/eval/<set>.csv, results/simcse/<set>.csv, results/llm/<set>.csv
Missing pieces are skipped, so you can run this at any stage.
Writes results/tables/main_results.csv/.tex and results/tables/own_breakdown.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stats_utils import bootstrap_ci

SETS = ["stsb_sample", "biosses", "own"]


def load_set(name):
    df = pd.read_csv(f"data/eval/{name}.csv")
    for kind, col in [("simcse", "simcse_sim"), ("llm", "llm_score")]:
        p = f"results/{kind}/{name}.csv"
        if os.path.exists(p):
            df = df.merge(pd.read_csv(p)[["pair_id", col]], on="pair_id", how="left")
    return df


def fmt(est, lo, hi):
    if est != est:                      # NaN: constant scores, nothing to correlate
        return "n/a"
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def corr_cell(df, a, b):
    if a not in df or b not in df:
        return ""
    d = df[[a, b]].dropna()
    if len(d) < 10:
        return ""
    return fmt(*bootstrap_ci(d[a], d[b]))


def main():
    os.makedirs("results/tables", exist_ok=True)
    rows = []
    for name in SETS:
        if not os.path.exists(f"data/eval/{name}.csv"):
            print("skip (no eval file):", name)
            continue
        df = load_set(name)
        row = {"set": name, "n": len(df),
               "simcse_vs_gold": corr_cell(df, "simcse_sim", "gold"),
               "llm_vs_gold": corr_cell(df, "llm_score", "gold")}
        if "simcse_sim" in df and "llm_score" in df:
            d = df[["simcse_sim", "llm_score"]].dropna()
            r = spearmanr(d.simcse_sim, d.llm_score)[0] if len(d) > 2 else float("nan")
            row["simcse_vs_llm"] = "n/a" if r != r else f"{r:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("results/tables/main_results.csv", index=False)
    print(out.to_string(index=False))

    tex = ["\\begin{tabular}{lrccc}", "\\toprule",
           "Set & $n$ & SimCSE vs.\\ gold & LLM vs.\\ gold & SimCSE vs.\\ LLM \\\\", "\\midrule"]
    for r in rows:
        tex.append(f"{r['set'].replace('_', ' ')} & {r['n']} & {r.get('simcse_vs_gold', '')} & "
                   f"{r.get('llm_vs_gold', '')} & {r.get('simcse_vs_llm', '')} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open("results/tables/main_results.tex", "w").write("\n".join(tex))

    # breakdown for your own dataset: bands and blind vs non-blind
    if os.path.exists("data/ratings_wide.csv") and os.path.exists("data/eval/own.csv"):
        own = load_set("own").merge(
            pd.read_csv("data/ratings_wide.csv")[["pair_id"] + [c for c in ["expected_band", "blind"]
                                                                  if c in pd.read_csv("data/ratings_wide.csv")]],
            on="pair_id", how="left")
        parts = []
        if "expected_band" in own:
            b = own.groupby("expected_band")["gold"].agg(["mean", "count"]).reset_index()
            b.insert(0, "split", "band")
            b = b.rename(columns={"expected_band": "group"})
            parts.append(b)
        if "blind" in own:
            for flag, g in own.groupby("blind"):
                for col in ["simcse_sim", "llm_score"]:
                    if col in g and g[col].notna().sum() > 5:
                        parts.append(pd.DataFrame([{"split": f"blind={flag}", "group": col, "count": len(g),
                                                    "mean": spearmanr(g[col], g["gold"])[0]}]))
        if parts:
            br = pd.concat(parts, ignore_index=True)
            br.to_csv("results/tables/own_breakdown.csv", index=False)
            print(br.round(3).to_string(index=False))
    if os.path.exists("results/tables/agreement_summary.csv"):
        print(pd.read_csv("results/tables/agreement_summary.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
