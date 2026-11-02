# Dataset documentation (datasheet) -- placeholder name: SyllabusSTS

> Fill every TODO before release. Do not copy licence terms from memory: read each source page.

**Purpose.** A small sentence-pair similarity dataset with human ratings, used to test how well
SimCSE transfers to course-outline text.

**Sources (public MIT OpenCourseWare syllabi).**
1. 6.006 Introduction to Algorithms (Spring 2020)
2. 7.016 Introductory Biology (Fall 2018)
3. 14.121 Microeconomic Theory I (Fall 2015)
4. 20.430J Fields, Forces, and Flows in Biological Systems (Fall 2015)
5. 20.320 Analysis of Biomolecular and Cellular Systems (Fall 2012)
URLs are in `src/build_dataset.py`. Date accessed: TODO. Licence stated on each page: TODO.

**Collection and processing.** Web pages fetched with `requests`; navigation/script elements removed;
text of `<p>` and `<li>` elements kept; rule-based sentence splitting; sentences under 6 words dropped
(`src/build_dataset.py`). `raw_texts.json` holds the cleaned text.

**Pairs.** 60 pairs: 30 "high" (same syllabus, at most 3 sentences apart), 15 "medium" (the two related
bioengineering syllabi), 15 "low" (unrelated subjects). The band is a sampling label, not a similarity rating.

**Annotation.** Scale 0-4 (`docs/annotation_guidelines.md`). Rater 1 is the project author; TODO other raters.
Pairs 0-29 were rated with the band visible, pairs 30-59 with it hidden (TODO confirm).
Agreement: TODO (see `results/tables/agreement*.csv`).

**Files.** `candidate_pairs.csv`, `annotations.csv` (rater 1), `rater2.csv` ..., `ratings_wide.csv`, `eval/`.

**Known limitations.** Very small; a single domain (course outlines); few high-similarity pairs;
few raters; rule-based sentence splitting; ratings reflect raters' judgement of text from one university.

**Personal information check.** TODO: scan sentences for staff names, emails and phone numbers before release.

**Intended use.** Research and teaching on sentence similarity. Not for evaluating or ranking people or courses.
