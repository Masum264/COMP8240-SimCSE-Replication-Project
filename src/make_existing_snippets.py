"""Write results/tables/snippets_existing.tex: one example row per existing dataset.

Reads the scored pair files in results/existing/ and picks fixed row numbers, so the
table is the same on every run. Long texts are cut at 100 characters.
Usage: python src/make_existing_snippets.py
"""
import pandas as pd

# (dataset name, file stem, row number, label text)
PICKS = [
    ("Quora", "quora", 10, "duplicate"),
    ("Quora", "quora", 7, "not duplicate"),
    ("BIOSSES", "biosses", 2, None),
    ("CodeSearchNet", "codesearch", 2, "match"),
    ("Amazon", "amazon", 26, "match"),
    ("Amazon", "amazon", 1051, "mismatch"),
    ("CUAD", "cuad", 4, "same category"),
    ("CUAD", "cuad", 848, "different category"),
]


def esc(t, n=100):
    t = " ".join(str(t).split())
    if len(t) > n:
        t = t[:n].rstrip() + "..."
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"),
                 ("^", r"\^{}")]:
        if a != "\\":
            t = t.replace(a, b)
    return t


rows = []
for name, stem, i, lab in PICKS:
    d = pd.read_csv(f"results/existing/{stem}_pairs_scored.csv")
    r = d.iloc[i]
    s1 = str(r.sentence1).split("\n")[0] + (" ..." if stem == "codesearch" else "") if stem == "codesearch" else r.sentence1
    s2 = str(r.sentence2).split("\n")[0]
    lab = lab if lab else f"{float(r.label):.1f}"
    rows.append(f"{name} & {esc(s1)} & {esc(s2)} & {lab} \\\\")

tex = ("\\setlength{\\tabcolsep}{4pt}\n"
       "\\begin{tabularx}{\\textwidth}{@{}l>{\\raggedright\\arraybackslash}X"
       ">{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{2.1cm}@{}}\n\\toprule\n"
       "Dataset & Text 1 & Text 2 & Label \\\\\n\\midrule\n" + "\n".join(rows)
       + "\n\\bottomrule\n\\end{tabularx}\n")
open("results/tables/snippets_existing.tex", "w").write(tex)
print(tex)
