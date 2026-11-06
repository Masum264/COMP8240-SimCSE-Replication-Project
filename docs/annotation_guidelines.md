# Annotation guidelines: sentence-pair similarity (0-4)

You will see pairs of sentences taken from university course outlines. For each pair, rate how
similar their **meaning** is. Use whole numbers only.

| Score | Meaning |
|---|---|
| 4 | **Same meaning.** The sentences say essentially the same thing. |
| 3 | **Mostly similar.** Similar meaning, but some important information differs. |
| 2 | **Somewhat related.** They share some details or a sub-topic, but are not equivalent. |
| 1 | **Barely related.** Same broad topic or domain only. |
| 0 | **Unrelated.** Different topics. |

Rules
- Judge meaning, not shared words. Two sentences can share many words and still mean different things.
- Ignore formatting, capitalisation and small grammatical slips.
- Rate each pair on its own. Do not try to guess how others rated it.
- Rows are in random order. Do not look anything up and do not discuss ratings with other raters until finished.

Illustrative examples (invented for these guidelines; not from the dataset)
- "Students submit weekly problem sets." / "Weekly problem sets are handed in by students." -> 4
- "Students submit weekly problem sets." / "Students submit weekly problem sets and a final report." -> 3
- "The course covers sorting algorithms." / "Students will analyse the running time of algorithms." -> 2
- "The course covers sorting algorithms." / "Grades are posted on the course website." -> 1
- "The course covers sorting algorithms." / "Enzymes lower the activation energy of reactions." -> 0

The scale is adapted from the 0-4 similarity scales used in semantic-textual-similarity datasets
such as BIOSSES (check that paper before citing it).
