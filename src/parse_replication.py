"""Parse evaluation.py outputs in results/replication/*.txt and compare with
Table 5 of Gao et al. (2021). Writes results/tables/replication.csv and .tex"""
import csv
import glob
import os
import re

COLS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STS-B", "SICK-R", "Avg."]
# Values transcribed from Table 5 of the SimCSE paper (Gao et al., 2021).
PAPER = {
    "sup-bert-base":      [75.30, 84.67, 80.19, 85.40, 80.82, 84.25, 80.39, 81.57],
    "unsup-bert-base":    [68.40, 82.41, 74.38, 80.91, 78.56, 76.85, 72.23, 76.25],
    "sup-roberta-base":   [76.53, 85.21, 80.95, 86.03, 82.57, 85.83, 80.50, 82.52],
    "unsup-roberta-base": [70.16, 81.77, 73.24, 81.36, 80.65, 80.22, 68.56, 76.57],
}
FLOAT = re.compile(r"-?\d+\.\d+")


def parse_sts_row(text):
    """Return the 8 numbers (7 STS tasks + average) from the first results table."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "STS12" in line and "STS13" in line:
            for nxt in lines[i + 1:i + 4]:
                vals = FLOAT.findall(nxt)
                if len(vals) == 8:
                    return [float(v) for v in vals]
    raise ValueError("no STS results table found")


def main(indir="results/replication", outdir="results/tables"):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(indir, "*.txt"))):
        key = os.path.splitext(os.path.basename(path))[0]
        if key not in PAPER:
            continue
        ours = parse_sts_row(open(path, encoding="utf-8").read())
        for col, o, p in zip(COLS, ours, PAPER[key]):
            rows.append({"model": key, "task": col, "ours": o, "paper": p,
                         "diff": round(o - p, 2)})
    with open(os.path.join(outdir, "replication.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task", "ours", "paper", "diff"])
        w.writeheader()
        w.writerows(rows)
    models = sorted({r["model"] for r in rows})
    tex = ["\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
           "Model & " + " & ".join(COLS) + " \\\\", "\\midrule"]
    for m in models:
        cells = [f"{r['ours']:.2f}/{r['paper']:.2f}" for r in rows if r["model"] == m]
        tex.append(m + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(outdir, "replication.tex"), "w").write("\n".join(tex))
    for r in rows:
        print(f"{r['model']:20s} {r['task']:7s} ours={r['ours']:6.2f} paper={r['paper']:6.2f} diff={r['diff']:+.2f}")
    return rows


if __name__ == "__main__":
    main()
