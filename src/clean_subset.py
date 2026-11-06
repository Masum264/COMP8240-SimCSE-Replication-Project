"""Flag rows whose sentences contain person names or reference lines, then repeat the main
comparison on the remaining ("clean") rows.

  python src/clean_subset.py

Writes data/row_flags.csv (pair_id, person, reference, dept_only, clean) and
results/tables/extra_clean_subset.csv. The flags come from keyword patterns, so they are a
screening tool and not a proof: read the flagged text yourself before relying on the counts.
  person     "Last, First" author lists, or Prof. / Dr. / Professor / Instructor / Teaching Assistant
  reference  ISBN, "Preview with Google Books", or an edition marker such as "3rd ed."
  dept_only  matches only a department name (needs reading by hand; kept in the clean subset)
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import load_set
from stats_utils import bootstrap_ci

FALSE_ALARMS = {"Field, Forces", "Python, Py", "Science, Analytics", "Calculus, Linear"}
NAME_LIST = re.compile(r"[A-Z][a-z]+, [A-Z][a-z]+(?:, [A-Z][a-z]+)*")
TITLE = re.compile(r"\bDr\.|\bProf\.|Professor|Instructor|Teaching Assistant")
REF = re.compile(r"ISBN|Preview with Google Books|\bed\.")
DEPT = re.compile(r"Electrical Engineering and Computer Science|Linguistics and Philosophy|Biological Engineering")


def sentence_flags(s):
    names = [m.group(0) for m in NAME_LIST.finditer(s)]
    person = any(n[:n.rfind(",")] not in FALSE_ALARMS and not any(n.startswith(f) for f in FALSE_ALARMS)
                 for n in names) or bool(TITLE.search(s))
    return person, bool(REF.search(s)), bool(DEPT.search(s))


def row_flags(candidates="data/candidate_pairs.csv"):
    df = pd.read_csv(candidates)
    out = []
    for pid, r in df.iterrows():
        f1, f2 = sentence_flags(r["sentence1"]), sentence_flags(r["sentence2"])
        person, ref, dept = f1[0] or f2[0], f1[1] or f2[1], f1[2] or f2[2]
        out.append({"pair_id": pid, "person": person, "reference": ref,
                    "dept_only": dept and not person and not ref, "clean": not person and not ref})
    return pd.DataFrame(out)


def main():
    flags = row_flags()
    flags.to_csv("data/row_flags.csv", index=False)
    print(f"rows: {len(flags)} | person: {flags.person.sum()} | reference: {flags.reference.sum()} | "
          f"person or reference: {(~flags.clean).sum()} | dept only: {flags.dept_only.sum()} | clean: {flags.clean.sum()}")
    df = load_set("own").merge(flags[["pair_id", "clean"]], on="pair_id")
    rows = []
    for label, sub in [("all rows", df), ("clean rows only", df[df.clean])]:
        d = sub[["simcse_sim", "llm_score", "gold"]].dropna().reset_index(drop=True)
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(5000):
            s = d.iloc[rng.integers(0, len(d), len(d))]
            diffs.append(spearmanr(s.llm_score, s.gold)[0] - spearmanr(s.simcse_sim, s.gold)[0])
        lo, hi = np.percentile([x for x in diffs if x == x], [2.5, 97.5])
        e1, l1, h1 = bootstrap_ci(d.simcse_sim, d.gold)
        e2, l2, h2 = bootstrap_ci(d.llm_score, d.gold)
        rows.append({"subset": label, "n": len(d), "simcse": f"{e1:.3f} [{l1:.3f}, {h1:.3f}]",
                     "judge": f"{e2:.3f} [{l2:.3f}, {h2:.3f}]",
                     "judge_minus_simcse": f"{e2 - e1:+.3f} [{lo:.3f}, {hi:.3f}]"})
    out = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    out.to_csv("results/tables/extra_clean_subset.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
