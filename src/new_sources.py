"""Eight additional MIT OpenCourseWare syllabus pages used to extend the dataset to 100 pairs.

These sources are DISJOINT from the five used for the first 60 pairs, so the extra pairs
contain sentences that never appear in the first batch. Fetch and inspect them with:

  python src/new_sources.py --out data/raw_texts_new.json

Band design for the new pairs (same logic as the first batch):
  high   = two sentences from the same syllabus, at most 3 sentences apart
  medium = two syllabi from the same broad field (RELATED)
  low    = two syllabi from unrelated fields (UNRELATED)
"""
import argparse
import json
import sys

NEW_SOURCES = {
    "intro_cs_2016": "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/",
    "intro_cs_2022": "https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/pages/syllabus/",
    "linear_algebra": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/syllabus/",
    "statistics": "https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/pages/syllabus/",
    "real_analysis": "https://ocw.mit.edu/courses/18-100c-real-analysis-fall-2012/pages/syllabus/",
    "computer_graphics": "https://ocw.mit.edu/courses/6-837-computer-graphics-fall-2012/pages/syllabus/",
    "theory_of_knowledge": "https://ocw.mit.edu/courses/24-211-theory-of-knowledge-spring-2014/pages/syllabus/",
    "education": "https://ocw.mit.edu/courses/11-125-introduction-to-education-understanding-and-evaluating-education-spring-2009/pages/syllabus/",
}

RELATED = [("intro_cs_2016", "intro_cs_2022"), ("linear_algebra", "statistics"),
           ("linear_algebra", "real_analysis"), ("intro_cs_2016", "computer_graphics")]
UNRELATED = [("theory_of_knowledge", "linear_algebra"), ("education", "intro_cs_2016"),
             ("theory_of_knowledge", "computer_graphics"), ("education", "statistics")]


def main():
    sys.path.insert(0, "src")
    from build_dataset import fetch_clean_text, split_into_sentences
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_texts_new.json")
    a = ap.parse_args()
    raw = {}
    for name, url in NEW_SOURCES.items():
        try:
            raw[name] = fetch_clean_text(url)
        except Exception as e:  # keep going, report at the end
            print(f"FAILED {name}: {e}")
    json.dump(raw, open(a.out, "w", encoding="utf-8"))
    total = 0
    for k, v in raw.items():
        n = len(split_into_sentences(v))
        total += n
        print(f"{k:22s} usable sentences: {n}")
    print("total usable sentences:", total, "->", a.out)


if __name__ == "__main__":
    main()
