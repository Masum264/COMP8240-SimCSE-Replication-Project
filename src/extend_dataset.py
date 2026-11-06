"""Extend data/candidate_pairs.csv from 60 to 100 pairs WITHOUT touching the first 60.

The 40 new pairs are drawn with a new seed from the same five saved syllabus texts
(data/raw_texts.json), using the same band design as build_dataset.py:
  20 high   (same syllabus, <=3 sentences apart: 4 per syllabus)
  10 medium (the two related bioengineering syllabi)
  10 low    (unrelated subjects: 4 + 3 + 3 over the three subject combinations)
Pairs already present (in either order), or with identical sentences, are skipped.
New rows get pair_id 60..99 (row order = pair_id, as everywhere else in the repo).

  python src/extend_dataset.py --raw data/raw_texts.json --candidates data/candidate_pairs.csv
"""
import argparse
import json
import random

import pandas as pd

from build_dataset import sample_across, sample_within, split_into_sentences


def draw(fn, n, seen, used, tries=200, reuse=False):
    """Call fn(k) until n pairs are collected whose two sentences are both unused
    (not in the existing pairs and not in earlier new pairs)."""
    out = []
    for _ in range(tries):
        for a, b in fn(n * 3):
            if a == b or a in used or b in used or (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            if not reuse:
                used.add(a)
                used.add(b)
            out.append((a, b))
            if len(out) == n:
                return out
    raise SystemExit("could not draw enough new unique pairs; try another --seed")


def extend(docs, existing, seed=2026, n_med=10, n_low=(4, 3, 3), target=100, allow_reuse=False):
    """Draw enough new pairs so that the file holds `target` DISTINCT pairs in total.
    The base design is 20 high / 10 medium / 10 low; any shortfall (because the old
    file contains repeated pairs) is made up with extra 'high' pairs."""
    random.seed(seed)
    distinct_old = len(existing.drop_duplicates(["sentence1", "sentence2"]))
    n_new = target - distinct_old
    base_high = n_new - n_med - sum(n_low)
    if base_high < len(docs):
        raise SystemExit(f"need {n_new} new pairs, too few for the design")
    per_doc = [base_high // len(docs) + (1 if i < base_high % len(docs) else 0) for i in range(len(docs))]
    seen = set(zip(existing["sentence1"], existing["sentence2"]))
    used = set() if allow_reuse else set(existing["sentence1"]) | set(existing["sentence2"])
    # allow_reuse: only whole pairs must be new; a sentence may appear in several pairs
    new = []
    for sents, nh in zip(docs.values(), per_doc):
        new += [(a, b, "high") for a, b in
                draw(lambda k, s=sents: sample_within(s, k, max_dist=3), nh, seen, used, reuse=allow_reuse)]
    new += [(a, b, "medium") for a, b in draw(
        lambda k: sample_across(docs["biological_systems"], docs["biomolecular_systems"], k), n_med, seen, used, reuse=allow_reuse)]
    combos = [("algorithms", "microeconomics"), ("algorithms", "biology"), ("microeconomics", "biology")]
    for (da, db), n in zip(combos, n_low):
        new += [(a, b, "low") for a, b in draw(lambda k, a_=da, b_=db: sample_across(docs[a_], docs[b_], k), n, seen, used, reuse=allow_reuse)]
    random.shuffle(new)
    return pd.DataFrame(new, columns=["sentence1", "sentence2", "expected_band"])


def extend_new_sources(docs, existing, seed=2026, target=100, allow_reuse_new=False):
    """Draw the extra pairs ONLY from new syllabi (docs). Sentences from the existing pairs
    are always excluded, so the new pairs never reuse an old sentence."""
    from new_sources import RELATED, UNRELATED
    random.seed(seed)
    seen = set(zip(existing["sentence1"], existing["sentence2"]))
    used = set(existing["sentence1"]) | set(existing["sentence2"])
    n_new = target - len(existing.drop_duplicates(["sentence1", "sentence2"]))
    n_med = n_low = n_new // 4
    n_high = n_new - n_med - n_low
    rel = [p for p in RELATED if p[0] in docs and p[1] in docs]
    unrel = [p for p in UNRELATED if p[0] in docs and p[1] in docs]
    if not rel or not unrel:
        raise SystemExit("not enough new syllabi were fetched for the related/unrelated bands")

    def split(total, k, weights=None):
        """Split `total` into k integer quotas (largest remainder), optionally weighted."""
        if weights is None:
            return [total // k + (1 if i < total % k else 0) for i in range(k)]
        tot = float(sum(weights))
        raw_q = [total * w / tot for w in weights]
        q = [int(x) for x in raw_q]
        for i in sorted(range(k), key=lambda i: raw_q[i] - q[i], reverse=True)[: total - sum(q)]:
            q[i] += 1
        return q

    new = []
    names = [k for k in docs if len(docs[k]) >= 12]   # tiny syllabi are used only for medium/low
    for name, n in zip(names, split(n_high, len(names), [len(docs[k]) for k in names])):
        new += [(a, b, "high") for a, b in draw(lambda k, s_=docs[name]: sample_within(s_, k, max_dist=3),
                                                n, seen, used, reuse=allow_reuse_new)]
    for (da, db), n in zip(rel, split(n_med, len(rel), [min(len(docs[a]), len(docs[b])) for a, b in rel])):
        new += [(a, b, "medium") for a, b in draw(lambda k, a_=da, b_=db: sample_across(docs[a_], docs[b_], k),
                                                  n, seen, used, reuse=allow_reuse_new)]
    for (da, db), n in zip(unrel, split(n_low, len(unrel), [min(len(docs[a]), len(docs[b])) for a, b in unrel])):
        new += [(a, b, "low") for a, b in draw(lambda k, a_=da, b_=db: sample_across(docs[a_], docs[b_], k),
                                               n, seen, used, reuse=allow_reuse_new)]
    random.shuffle(new)
    return pd.DataFrame(new, columns=["sentence1", "sentence2", "expected_band"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_texts.json")
    ap.add_argument("--candidates", default="data/candidate_pairs.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--target", type=int, default=100, help="total DISTINCT pairs wanted")
    ap.add_argument("--allow_reuse", action="store_true", help="only require new PAIRS (sentences may repeat); needed when the syllabi are small")
    ap.add_argument("--raw_new", help="JSON of NEW syllabi (from new_sources.py); pairs are drawn only from these")
    ap.add_argument("--allow_reuse_new", action="store_true", help="with --raw_new: a new sentence may appear in more than one new pair")
    ap.add_argument("--expect", type=int, default=60, help="number of existing pairs that must be kept")
    a = ap.parse_args()
    raw = json.load(open(a.raw_new or a.raw, encoding="utf-8"))
    docs = {k: split_into_sentences(v) for k, v in raw.items()}
    old = pd.read_csv(a.candidates)
    if len(old) != a.expect:
        raise SystemExit(f"expected {a.expect} existing pairs, found {len(old)} (already extended?)")
    old.to_csv(a.candidates.replace(".csv", "_first60_backup.csv"), index=False)
    if a.raw_new:
        new = extend_new_sources(docs, old, a.seed, a.target, a.allow_reuse_new)
        a.allow_reuse = a.allow_reuse_new   # post-checks below
    else:
        new = extend(docs, old, a.seed, target=a.target, allow_reuse=a.allow_reuse)
    full = pd.concat([old, new], ignore_index=True)
    assert (full.iloc[:len(old)].values == old.values).all()
    if not a.allow_reuse:
        used_old = set(old["sentence1"]) | set(old["sentence2"])
        assert not (set(new["sentence1"]) | set(new["sentence2"])) & used_old, "a new pair reuses an old sentence"
        assert len(set(new["sentence1"]) | set(new["sentence2"])) == 2 * len(new), "a sentence repeats within the new pairs"
    allsents = set(full["sentence1"]) | set(full["sentence2"])
    print("distinct sentences across all pairs:", len(allsents))
    full.to_csv(a.candidates, index=False)
    print("distinct pairs now:", len(full.drop_duplicates(["sentence1", "sentence2"])))
    print(f"{a.candidates}: {len(old)} -> {len(full)} pairs (new ids {len(old)}-{len(full)-1})")
    print(full.iloc[len(old):]["expected_band"].value_counts().to_string())
