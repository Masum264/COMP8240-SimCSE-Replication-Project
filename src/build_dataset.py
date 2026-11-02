"""Build candidate sentence pairs from five MIT OpenCourseWare syllabi.

NOTE: the pairs actually annotated for the report are in data/candidate_pairs.csv
(copied from the project's Google Drive). This script documents and re-creates the
procedure; by default it writes to data/candidate_pairs_regenerated.csv and never
overwrites the original. Use --verify to check that regeneration matches.

  python src/build_dataset.py --from_json data/raw_texts.json --verify data/candidate_pairs.csv
"""
import argparse
import json
import os
import random
import re

import pandas as pd

SOURCES = {
    "algorithms": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/",
    "biology": "https://ocw.mit.edu/courses/7-016-introductory-biology-fall-2018/pages/syllabus",
    "microeconomics": "https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/pages/syllabus",
    "biological_systems": "https://ocw.mit.edu/courses/20-430j-fields-forces-and-flows-in-biological-systems-fall-2015/pages/syllabus",
    "biomolecular_systems": "https://ocw.mit.edu/courses/20-320-analysis-of-biomolecular-and-cellular-systems-fall-2012/pages/syllabus",
}


def fetch_clean_text(url):
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return " ".join(el.get_text(" ", strip=True) for el in soup.find_all(["p", "li"]))


def split_into_sentences(text, min_words=6):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sents = [s.strip().replace("\n", " ") for s in sents]
    return [s for s in sents if len(s.split()) >= min_words]


def sample_within(sentences, n, max_dist=None, min_dist=0):
    pairs, attempts = [], 0
    while len(pairs) < n and attempts < n * 20:
        i, j = random.sample(range(len(sentences)), 2)
        dist = abs(i - j)
        if dist >= min_dist and (max_dist is None or dist <= max_dist):
            pairs.append((sentences[i], sentences[j]))
        attempts += 1
    return pairs


def sample_across(a, b, n):
    return [(random.choice(a), random.choice(b)) for _ in range(n)]


def build_pairs(doc_sentences, seed=42):
    """30 'high' (same syllabus, <=3 sentences apart), 15 'medium' (two related
    bioengineering syllabi), 15 'low' (unrelated subjects)."""
    random.seed(seed)
    pairs = []
    for sents in doc_sentences.values():
        pairs += [(a, b, "high") for a, b in sample_within(sents, 6, max_dist=3)]
    pairs += [(a, b, "medium") for a, b in sample_across(
        doc_sentences["biological_systems"], doc_sentences["biomolecular_systems"], 15)]
    for da, db in [("algorithms", "microeconomics"), ("algorithms", "biology"),
                   ("microeconomics", "biology")]:
        pairs += [(a, b, "low") for a, b in sample_across(doc_sentences[da], doc_sentences[db], 5)]
    random.shuffle(pairs)
    return pd.DataFrame(pairs, columns=["sentence1", "sentence2", "expected_band"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_json", help="use saved raw texts instead of fetching")
    ap.add_argument("--raw_out", default="data/raw_texts.json")
    ap.add_argument("--pairs_out", default="data/candidate_pairs_regenerated.csv")
    ap.add_argument("--verify", help="existing candidate_pairs.csv to compare against")
    args = ap.parse_args()

    if args.from_json:
        raw = json.load(open(args.from_json, encoding="utf-8"))
    else:
        raw = {}
        for name, url in SOURCES.items():
            print("fetching", name)
            raw[name] = fetch_clean_text(url)
        if os.path.exists(args.raw_out):
            raise SystemExit(f"{args.raw_out} exists; refusing to overwrite (use --from_json)")
        json.dump(raw, open(args.raw_out, "w", encoding="utf-8"))
    docs = {k: split_into_sentences(v) for k, v in raw.items()}
    for k, v in docs.items():
        print(f"{k}: {len(v)} sentences")
    df = build_pairs(docs)
    df.to_csv(args.pairs_out, index=False)
    print(f"wrote {len(df)} pairs to {args.pairs_out}")
    print(df["expected_band"].value_counts().to_string())
    if args.verify:
        old = pd.read_csv(args.verify)
        same = (len(old) == len(df)
                and (old["sentence1"].values == df["sentence1"].values).all()
                and (old["sentence2"].values == df["sentence2"].values).all())
        print("regenerated pairs IDENTICAL to", args.verify if same else "-- DIFFERENT from " + args.verify)


if __name__ == "__main__":
    main()
