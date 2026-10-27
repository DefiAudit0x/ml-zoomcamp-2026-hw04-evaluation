# ML Zoomcamp 2026 — Homework 4

**Evaluation Metrics for Classification** — reproducible solution using the pinned 2026 lead-scoring dataset.

## Answers

| Question | Result | Selection |
|---|---:|---|
| Q1 — Highest single-feature ROC AUC | `lead_score` (0.788225) | `lead_score` |
| Q2 — Validation ROC AUC | 0.731822 | 0.732 |
| Q3 — Closest precision/recall threshold | 0.63 | 0.63 |
| Q4 — Threshold with maximum F1 | 0.41 (F1 = 0.757616) | 0.41 |
| Q5 — Standard deviation of 5-fold AUC | 0.006853 | 0.007 |
| Q6 — Best C by mean 5-fold ROC AUC | 0.001 (mean = 0.749172) | 0.001 |

The full precision/recall and F1 threshold curves are in `figures/`. Detailed values are also stored in `results.json`.

## Run it

Use Python 3.11 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python solution.py
```

The script writes `results.json` and recreates the figures. Open `hw04_solution.ipynb` in Jupyter to review the same run with the answer summary and charts.

## Dataset and method

The dataset is the course's pinned 2026 release: [course_lead_scoring_2026.csv](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv). It contains 5,000 rows and the target `converted`. Its SHA-256 is `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9`.

The notebook and script follow the [official Homework 4 instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md): categorical missing values become `NA`, numeric missing values become `0.0`; the exact train/validation/test split uses `test_size=0.2` and then `test_size=0.25`, both with `random_state=1`. Model evaluation uses `DictVectorizer` and `LogisticRegression(solver="liblinear", C=C, max_iter=1000)`. `random_state=42` is set for repeatable liblinear fitting, matching the course release checker.

The 5-fold validation and C comparison use `KFold(n_splits=5, shuffle=True, random_state=1)` on the full 80% training portion. The held-out test split is reserved and is not used in the homework questions.

## Sources

- [Homework 4 — ML Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04)
- [Official homework instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md)
- [Official 2026 data release and checksums](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/data)
