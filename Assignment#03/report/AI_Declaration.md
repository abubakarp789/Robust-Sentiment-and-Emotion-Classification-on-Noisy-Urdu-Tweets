# AI Usage Declaration — Assignment 3 (Milestone 3)

**Project:** Robust Sentiment and Emotion Classification on Noisy Urdu Tweets
**Course:** CSC-355 Natural Language Processing
**University:** Namal University Mianwali
**Authors:** Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)
**Date:** 2026-05-16

---

## 1. AI tools used

| Tool | Provider | Version | Purpose |
|---|---|---|---|
| Claude Code | Anthropic | Opus 4.7 (claude-opus-4-7) | Coding assistant for module scaffolding, notebook generation, and first-draft report writing. |

## 2. What the AI did

- Extracted the eight-step preprocessing pipeline from Milestone 1's notebook into a reusable, importable `preprocessing.py` module. The transformations themselves were already designed and justified by the authors in Milestone 1; the AI assisted by packaging them as a clean Python module with type hints and tests.
- Generated the **scaffolding** of the three Jupyter notebooks (cell structure, section headings, helper imports). The authors then executed each cell, inspected the outputs, fixed bugs (e.g. the label-canonicalisation issue described in Section 2.2 of the report), and added domain-specific commentary.
- Wrote the first draft of the `04_architecture_math.md` document. Equations were verified against Goodfellow et al. (2016) and the original papers cited in the references.
- Wrote the first draft of this technical report. The authors edited it for accuracy, added the empirical numbers from their own training runs, and rewrote the discussion sections.
- Suggested hyper-parameter defaults in `config.py`. The final values were chosen by the authors based on local benchmarking.

## 3. What the AI did *not* do

- The AI did **not** select the dataset, choose the labelling strategy, decide on the model lineup, or make architectural choices unilaterally. Every decision was made by the authors and the AI was instructed accordingly.
- The AI did **not** run the model training or generate the final numbers in the leaderboard tables; those were produced by the authors executing the notebooks on their own hardware.
- No private or unpublished data was shared with the AI. The only inputs were (i) source code authored by us and (ii) the public structure of the SentiUrdu-1M dataset (column names, label set, dataset DOI).

## 4. Authors' responsibility statement

We have read every line of code and every section of this report and the architecture/math document. We accept responsibility for all design decisions, all numerical results, and the technical correctness of the implementation. Errors, if any, are ours, not the AI's.

—
*Raqib Hayat (NUM-BSCS-2022-40)*
*Abu Bakar (NUM-BSCS-2022-41)*
