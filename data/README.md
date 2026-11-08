# Dataset documentation (datasheet): SyllabusSTS (working name)

**Purpose.** A small sentence-pair similarity dataset with human ratings, used to test how well SimCSE and an
LLM judge transfer to course-outline text.

**Size.** 102 rows = 100 distinct sentence pairs (two pairs were drawn twice in the first batch and are kept as
separate rows, which gives a free test-retest check). About 141 distinct sentences.

**Sources (public MIT OpenCourseWare syllabus pages).**
- Batch 1 (pair ids 0-59), five syllabi: 6.006 Introduction to Algorithms (Spring 2020), 7.016 Introductory Biology
  (Fall 2018), 14.121 Microeconomic Theory I (Fall 2015), 20.430J Fields, Forces, and Flows in Biological Systems
  (Fall 2015), 20.320 Analysis of Biomolecular and Cellular Systems (Fall 2012).
- Batch 2 (pair ids 60-101), eight further syllabi: 6.0001 Intro to CS and Programming in Python (Fall 2016),
  6.100L Intro to CS and Programming using Python (Fall 2022), 18.06 Linear Algebra (Spring 2010),
  18.650 Statistics for Applications (Fall 2016), 18.100C Real Analysis (Fall 2012), 6.837 Computer Graphics (Fall 2012),
  24.211 Theory of Knowledge (Spring 2014), 11.125 Introduction to Education (Spring 2009).
- URLs are in `src/build_dataset.py` (batch 1) and `src/new_sources.py` (batch 2).
- Date accessed: batch 1: 2 Oct 2026, batch 2: 5 Nov 2026. Licence stated on each page: Creative Commons
  CC BY-NC-SA 4.0 (link in the page footer).

**Collection and processing.** Pages fetched with `requests`; script, style and navigation elements removed; text of
`<p>` and `<li>` elements kept; rule-based sentence splitting; sentences under 6 words dropped
(`src/build_dataset.py`). `raw_texts.json` and `raw_texts_new.json` hold the cleaned text of the two batches.

**Pairs.** Each pair carries an expected band, which is a sampling label and not a similarity rating.
Batch 1 (60 pairs): 30 high (same syllabus, at most 3 sentences apart), 15 medium (two related bioengineering
syllabi), 15 low (unrelated subjects). Batch 2 (42 rows, `src/extend_dataset.py`, no sentence shared with batch 1):
22 high, 10 medium, 10 low, with medium drawn from syllabi of the same broad field and low from unrelated fields.
Totals: 52 high, 25 medium, 25 low rows.

**Annotation.** Scale 0-4 (`docs/annotation_guidelines.md`). Three raters: the project author and two volunteer peers,
unpaid, told the purpose of the study. The gold score is the mean of the three ratings.
Author's ratings of pairs 0-29 were made with the expected band visible; all other ratings by all raters were blind.
Agreement (n = 102 rows): mean pairwise Spearman 0.73, mean quadratic-weighted kappa 0.69, Krippendorff's ordinal
alpha 0.65 (`results/tables/agreement*.csv`).

**Files.** `candidate_pairs.csv` (sentence pairs and expected band; the row number is the pair id),
`annotations.csv` (rater 1, with the sentences), `rater2.csv` and `rater3.csv` (pair id and score only),
`ratings_wide.csv` (all three ratings, their mean as `gold`, and the sentences), `raw_texts.json` and
`raw_texts_new.json` (cleaned syllabus text of batch 1 and batch 2), `row_flags.csv` (see below), and
`eval/own.csv` (this dataset in the format the scoring scripts read). `eval/biosses.csv` and
`eval/stsb_sample.csv` are not distributed: run `python src/make_eval_sets.py` to regenerate them from their
public sources.

**Known limitations.** Small; a single genre (course outlines from one university); mean ratings are low (about 1.0
to 1.4 on a 0-4 scale) so there are few near-paraphrases; three raters, two of them peers of the author; rule-based
sentence splitting; "high" pairs are neighbouring sentences, which are often not similar in meaning.

**Personal information check.** A keyword scan (`src/clean_subset.py`) of all 102 rows found likely person
names in 22 rows (textbook author lists, and titles such as Prof. or Dr.) and ISBN or edition reference lines in
13 rows, 29 rows in total. A further 16 rows match only a department name and have not been read by hand. The scan
is a screen, not a guarantee. A separate pattern check found no email addresses, web links or phone-number formats.
The names come from public course pages where instructors and authors are credited. `row_flags.csv` marks each row
(`person`, `reference`, `dept_only`, `clean`) so that users can drop them. The text of all rows, including the flagged
ones, is released (see Release and removal below).

**Data quality.** The text was fetched without forcing an encoding, so some batch 2 sentences contain broken
characters, and the sentence splitter produced some fragments and page titles. Raters saw the text as stored.

**Intended use.** Research and teaching on sentence similarity. Not for evaluating or ranking people or courses.

**Release and removal.** The text is released as it is, with attribution to MIT OpenCourseWare, under
CC BY-NC-SA 4.0, for non-commercial use only. The ratings are released without rater names (rater 1, 2, 3).
`row_flags.csv` marks the 29 rows that contain likely person names or reference lines. To request removal of a row,
open an issue on this repository.
