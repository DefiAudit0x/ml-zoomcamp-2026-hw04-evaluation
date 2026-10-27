# Homework 4 — Answers for review

Source: [ML Zoomcamp 2026 Homework 4](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04) and its [official instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md).

1. **ROC AUC feature importance:** `lead_score` — AUC 0.788225 (direction-adjusted). Other values: `interaction_count` 0.764920; `number_of_courses_viewed` 0.722989; `annual_income` 0.608198.
2. **Model AUC on validation:** 0.731822 → choose **0.732**.
3. **Precision/recall closest threshold:** 0.63 (precision 0.7328, recall 0.7253; absolute gap 0.0075) → choose **0.63**.
4. **Threshold maximizing F1:** 0.41 (F1 0.757616) → choose **0.41**.
5. **5-fold CV standard deviation:** 0.006853 → choose **0.007**.
6. **Best C:** 0.001 (mean fold AUC 0.749172; std 0.009996) → choose **0.001**.

## Reproducibility details

- Dataset: official pinned 2026 lead-scoring release, 5,000 rows.
- Split sizes: train 3,000; validation 1,000; held-out test 1,000.
- Missing categorical values were filled with `NA`; missing numeric values with `0.0`.
- The assignment's split calls, feature encoding, model settings, threshold grid, and 5-fold setup were followed. `random_state=42` is set on logistic regression to match the release checker and make fitting repeatable.
- The held-out test split was not used to tune or report the homework answers.

## Optional learning-in-public draft (not posted)

> Completed Module 4 of #mlzoomcamp 2026: I compared ROC AUC with threshold-based precision, recall and F1, then used 5-fold cross-validation to choose regularization. I learned that the decision threshold should reflect the precision/recall trade-off—not default automatically to 0.5. @Alexey Grigorev

The homework's separate course-form submission and any public post have not been submitted or published.
