# Similarity / Plagiarism Report — Assignment 3 (Milestone 3)

**Project:** Robust Sentiment and Emotion Classification on Noisy Urdu Tweets
**Course:** CSC-355 Natural Language Processing
**University:** Namal University Mianwali
**Authors:** Raqib Hayat (NUM-BSCS-2022-40) and Abu Bakar (NUM-BSCS-2022-41)
**Date:** 2026-05-16

---

## 1. Tool used

The technical report and the architecture / mathematical-modelling document were
checked for similarity using the institutional plagiarism-detection service made
available to Namal University students:

| Tool        | Provider             | Version / access     |
|-------------|----------------------|----------------------|
| Turnitin    | Turnitin LLC         | Namal LMS submission |

(If the institution's Turnitin licence is unavailable at submission time, the
authors will substitute the result from the free `https://www.duplichecker.com/`
or `https://plagiarismdetector.net/` services; the reported numbers below will
be updated in the final submission.)

## 2. Documents submitted to the checker

| File                                       | What it contains                              |
|--------------------------------------------|------------------------------------------------|
| `report/Assignment3_TechnicalReport.docx`  | The compiled technical report                  |
| `04_architecture_math.md`                  | Architecture + mathematical-modelling document |
| `report/AI_Declaration.md`                 | AI-usage declaration                           |

Code files (`.py`, `.ipynb`) are not submitted to the prose-similarity checker
because the equation/code overlap with standard library / framework
documentation would produce uninformative numbers; instead, authorship of the
code is established by the commit history in the project's Git repository.

## 3. Reported similarity (to be filled by authors at final submission)

| Document                                    | Overall similarity % | Notes                                                  |
|---------------------------------------------|----------------------|--------------------------------------------------------|
| Assignment3_TechnicalReport.docx            | _to be filled_       | Verbatim formula snippets and cited paper titles excluded |
| 04_architecture_math.md                     | _to be filled_       | Standard equations (LSTM gates, attention, AdamW)      |
| AI_Declaration.md                           | _to be filled_       | Short document, expected ≤ 5 %                         |

**Acceptable threshold per departmental policy:** overall similarity ≤ 20 %,
with no single source contributing more than 5 %.

## 4. What sources of overlap are expected and why they are *not* plagiarism

- **Mathematical formulas** (LSTM gate equations, scaled dot-product attention,
  TF-IDF weighting, AdamW update, softmax cross-entropy) appear in essentially
  the same notation in every introductory deep-learning textbook (e.g.,
  Goodfellow et al., 2016) and in the original papers we cite. We *do* cite the
  original sources in Section 10 of the technical report.
- **Standard package names and API references** (e.g. `sklearn.feature_extraction.text.TfidfVectorizer`,
  `transformers.AutoModelForSequenceClassification`) will produce trivial
  overlap with public documentation; this is not plagiarism.
- **Dataset description** of SentiUrdu-1M is paraphrased from the Mendeley
  Data record and the dataset's accompanying paper (Ali et al., 2023). The
  paper is cited and direct quotations are avoided.

## 5. What was checked separately by hand

- All prose paragraphs in Sections 1, 2, 7, 8, 9 of the technical report were
  written from scratch for this milestone; no copy-paste from internet sources.
- All experimental numbers in Section 7 originate from the authors' own
  training runs (recorded in `outputs/results/leaderboard_{sentiment,emotion}.csv`
  and `outputs/results/pred_*.csv`).
- The "Common error patterns" examples in Section 8.2 are drawn from
  misclassified test rows printed by `03_evaluation.ipynb` and not from any
  external Urdu-NLP paper.

## 6. Author attestation

We attest that the prose in the technical report, the architecture
mathematical-modelling document, and the AI usage declaration is our own work
except where explicitly cited. We accept responsibility for any unintentional
overlap flagged by the similarity tool and will revise the prose if the
reported numbers exceed the departmental threshold.

—
*Raqib Hayat (NUM-BSCS-2022-40)*
*Abu Bakar (NUM-BSCS-2022-41)*
