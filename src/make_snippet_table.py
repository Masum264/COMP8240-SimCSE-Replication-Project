"""Write results/tables/snippets_own.tex: a few real pairs per expected band with their gold score.

  python src/make_snippet_table.py            # needs data/ratings_wide.csv
Short pairs only (both sentences <= --maxlen characters) so the table fits a two-column page.
"""
import argparse
import re

import pandas as pd

ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


BAD = re.compile(r"\bDr\.|\bProf\.|ISBN|\[|\]|@|https?://|\bet al\b|\b[A-Z]\.$|\bed\.|\s\.(\s|$)|[A-Z][a-z]+, [A-Z][a-z]+|'s [A-Z]")
ALLOW = {"MATLAB", "Python", "Linux", "Athena", "Piazza"}   # capitalised words that are not person names


def has_name_like(s):
    """True if a capitalised word appears after the first word (names, titles, places)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)[1:]
    return any(w[0].isupper() and w[1:].islower() and w not in ALLOW and w not in {"I"} for w in words)


def fix_text(s):
    """Repair UTF-8 text that was decoded as Latin-1, then keep plain ASCII for LaTeX."""
    s = str(s)
    try:
        s = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            s = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    s = s.replace("\u2014", " -- ").replace("\u2013", "--").replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return s.encode("ascii", "ignore").decode("ascii")


def usable(s, minlen):
    s = fix_text(s)
    return len(s) >= minlen and s.rstrip().endswith((".", "?", "!")) and not BAD.search(s) and not has_name_like(s) and s.count(" ") >= 5


def esc(s):
    return "".join(ESC.get(c, c) for c in str(s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/ratings_wide.csv")
    ap.add_argument("--out", default="results/tables/snippets_own.tex")
    ap.add_argument("--per_band", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=95)
    ap.add_argument("--minlen", type=int, default=35)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    w = pd.read_csv(a.ratings)
    w = w[(w.sentence1.str.len() <= a.maxlen) & (w.sentence2.str.len() <= a.maxlen)]
    w = w[[usable(x, a.minlen) and usable(y, a.minlen) for x, y in zip(w.sentence1, w.sentence2)]]
    rows = []
    for band in ["high", "medium", "low"]:
        g = w[w.expected_band == band]
        g = g.sample(n=min(a.per_band, len(g)), random_state=a.seed)
        for _, r in g.iterrows():
            rows.append(f"{band} & {esc(fix_text(r.sentence1))} & {esc(fix_text(r.sentence2))} & {r.gold:.2f} \\\\")
    body = "\n".join(rows)
    tex = ("\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}p{1.3cm}>{\\raggedright\\arraybackslash}p{6.3cm}>{\\raggedright\\arraybackslash}p{6.3cm}r@{}}\n\\toprule\nBand & Sentence 1 & Sentence 2 & Gold \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(a.out, "w", encoding="utf-8").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
