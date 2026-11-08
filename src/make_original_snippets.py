"""Write results/tables/snippets_orig.tex: one real pair from each of the seven STS evaluation sets.

  python src/make_original_snippets.py --root $THIRD_PARTY/SimCSE/SentEval/data/downstream

Reads the files that the authors' download script puts under SentEval/data/downstream
(STS12-16 input and gold-score files, STSBenchmark/sts-test.csv, SICK_test_annotated.txt).
Picks the first pair with two clean sentences of 25 to 80 characters, so the choice is deterministic.
"""
import argparse
import glob
import os
import re

from make_snippet_table import esc, fix_text

OK = re.compile(r"^[A-Za-z0-9 ,.'\"?!:;()-]+$")


def clean(s):
    s = fix_text(s).strip()
    return s if 25 <= len(s) <= 80 and OK.match(s) else None


def first_good(pairs):
    for s1, s2, score in pairs:
        a, b = clean(s1), clean(s2)
        if a and b and a != b:
            return a, b, score
    return None


def sts_year(root, year):
    d = os.path.join(root, "STS", f"STS{year}-en-test")
    for inp in sorted(glob.glob(os.path.join(d, "STS.input.*.txt"))):
        gs = inp.replace("STS.input.", "STS.gs.")
        if not os.path.exists(gs):
            continue
        lines = open(inp, encoding="utf-8", errors="ignore").read().splitlines()
        golds = open(gs, encoding="utf-8", errors="ignore").read().splitlines()
        rows = []
        for l, g in zip(lines, golds):
            parts = l.split("\t")
            if len(parts) >= 2 and g.strip():
                rows.append((parts[0], parts[1], float(g)))
        hit = first_good(rows)
        if hit:
            return hit
    return None


def stsb(root):
    f = os.path.join(root, "STS", "STSBenchmark", "sts-test.csv")
    rows = []
    for l in open(f, encoding="utf-8", errors="ignore"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 7:
            rows.append((p[5], p[6], float(p[4])))
    return first_good(rows)


def sick(root):
    for f in glob.glob(os.path.join(root, "SICK", "SICK_test_annotated.txt")):
        rows = []
        for l in list(open(f, encoding="utf-8", errors="ignore"))[1:]:
            p = l.rstrip("\n").split("\t")
            if len(p) >= 4:
                rows.append((p[1], p[2], float(p[3])))
        return first_good(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("THIRD_PARTY", "third_party"),
                                                    "SimCSE/SentEval/data/downstream"))
    ap.add_argument("--out", default="results/tables/snippets_orig.tex")
    a = ap.parse_args()
    sets = [(f"STS{y}", lambda y=y: sts_year(a.root, y)) for y in (12, 13, 14, 15, 16)]
    sets += [("STS-B", lambda: stsb(a.root)), ("SICK-R", lambda: sick(a.root))]
    rows = []
    for name, fn in sets:
        hit = fn()
        if hit is None:
            print(f"no usable pair found for {name} under {a.root}")
            continue
        rows.append(f"{name} & {esc(hit[0])} & {esc(hit[1])} & {hit[2]:.1f} \\\\")
    tex = ("\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{@{}p{1.4cm}>{\\raggedright\\arraybackslash}p{6.2cm}>{\\raggedright\\arraybackslash}p{6.2cm}r@{}}\n"
           "\\toprule\nSet & Sentence 1 & Sentence 2 & Gold \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(tex)
    print(tex)


if __name__ == "__main__":
    main()
