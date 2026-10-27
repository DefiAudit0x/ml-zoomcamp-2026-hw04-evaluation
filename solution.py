"""Reproducible solution for ML Zoomcamp 2026 Homework 4.

Run from the project root:
    python solution.py

The script uses the pinned 2026 course dataset, the required 60/20/20 split,
DictVectorizer, LogisticRegression, threshold metrics, and 5-fold CV.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "course_lead_scoring_2026.csv"
FIGURES_DIR = ROOT / "figures"
TARGET = "converted"
CATEGORICAL_COLUMNS = ["lead_source", "industry", "employment_status", "location"]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Separate target and fill missing feature values as specified by the course."""
    features = df.drop(columns=[TARGET]).copy()
    # Preserve object dtype across pandas versions (pandas 3 infers string dtype).
    categorical = [c for c in CATEGORICAL_COLUMNS if c in features.columns]
    numerical = [c for c in features.columns if c not in categorical]
    features[categorical] = features[categorical].astype("object").fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)
    return features, categorical, numerical


def fit_model(
    train_x: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    validation_x: pd.DataFrame,
    c: float = 1.0,
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit DictVectorizer and logistic regression on one train/validation split."""
    dv = DictVectorizer(sparse=False)
    train_matrix = dv.fit_transform(train_x.to_dict(orient="records"))
    validation_matrix = dv.transform(validation_x.to_dict(orient="records"))
    model = LogisticRegression(
        solver="liblinear", C=c, max_iter=1000, random_state=42
    )
    model.fit(train_matrix, train_y)
    return model, validation_matrix


def main() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. See README.md for the official source."
        )

    # Explicitly read categoricals as object to keep feature handling stable.
    df = pd.read_csv(DATA_PATH, dtype={c: "object" for c in CATEGORICAL_COLUMNS})
    features, categorical, numerical = prepare_features(df)
    prepared = features.assign(**{TARGET: df[TARGET].to_numpy()})

    # Exact train/validation/test calls from the 2026 homework instructions.
    df_full_train, df_test = train_test_split(
        prepared, test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )
    y_train = df_train.pop(TARGET)
    y_val = df_val.pop(TARGET)

    # Q1: AUC for each numerical feature on the training set; orient above 0.5.
    feature_aucs = {}
    for column in numerical:
        auc = roc_auc_score(y_train, df_train[column])
        feature_aucs[column] = max(auc, 1.0 - auc)
    q1 = max(
        ["lead_score", "number_of_courses_viewed", "interaction_count", "annual_income"],
        key=feature_aucs.get,
    )

    # Q2: Validation ROC AUC.
    model, validation_matrix = fit_model(df_train, y_train, df_val)
    probabilities = model.predict_proba(validation_matrix)[:, 1]
    validation_auc = roc_auc_score(y_val, probabilities)

    # Q3/Q4: precision, recall, and F1 at the specified thresholds.
    thresholds = np.arange(0.0, 1.0, 0.01)
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(y_val, predicted, zero_division=0)
        recall = recall_score(y_val, predicted, zero_division=0)
        f1 = f1_score(y_val, predicted, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    metrics = pd.DataFrame(rows)
    eligible = metrics[(metrics["precision"] != 0) | (metrics["recall"] != 0)]
    q3_row = eligible.iloc[(eligible["precision"] - eligible["recall"]).abs().argmin()]
    q3 = round(float(q3_row["threshold"]), 2)
    q4_row = metrics.iloc[metrics["f1"].to_numpy().argmax()]
    q4 = round(float(q4_row["threshold"]), 2)

    # Q5: AUC mean and population standard deviation across five folds.
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_scores = []
    for fold_train_idx, fold_validation_idx in cv.split(df_full_train):
        fold_train = df_full_train.iloc[fold_train_idx].copy()
        fold_validation = df_full_train.iloc[fold_validation_idx].copy()
        fold_train_y = fold_train.pop(TARGET)
        fold_validation_y = fold_validation.pop(TARGET)
        fold_model, fold_matrix = fit_model(
            fold_train, fold_train_y, fold_validation
        )
        cv_scores.append(
            roc_auc_score(
                fold_validation_y, fold_model.predict_proba(fold_matrix)[:, 1]
            )
        )
    cv_std = float(np.std(cv_scores))

    # Q6: Compare the requested C values using the same folds.
    tuning = {}
    for c in [1e-6, 1e-3, 1.0]:
        scores = []
        for fold_train_idx, fold_validation_idx in cv.split(df_full_train):
            fold_train = df_full_train.iloc[fold_train_idx].copy()
            fold_validation = df_full_train.iloc[fold_validation_idx].copy()
            fold_train_y = fold_train.pop(TARGET)
            fold_validation_y = fold_validation.pop(TARGET)
            fold_model, fold_matrix = fit_model(
                fold_train, fold_train_y, fold_validation, c=c
            )
            scores.append(
                roc_auc_score(
                    fold_validation_y, fold_model.predict_proba(fold_matrix)[:, 1]
                )
            )
        tuning[c] = {"mean_auc": float(np.mean(scores)), "std_auc": float(np.std(scores))}
    q6 = max(
        tuning,
        key=lambda c: (round(tuning[c]["mean_auc"], 3), -round(tuning[c]["std_auc"], 3), -c),
    )

    # Save the precision/recall and F1 curves requested in the homework.
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["threshold"], metrics["precision"], label="Precision", lw=2)
    ax.plot(metrics["threshold"], metrics["recall"], label="Recall", lw=2)
    ax.axvline(q3, color="black", linestyle="--", alpha=0.6, label=f"Closest intersection: {q3:.2f}")
    ax.set(xlabel="Classification threshold", ylabel="Score", title="Precision and recall by threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_by_threshold.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["threshold"], metrics["f1"], color="#7a3db8", lw=2)
    ax.axvline(q4, color="black", linestyle="--", alpha=0.6, label=f"Maximum F1: {q4:.2f}")
    ax.scatter([q4], [q4_row["f1"]], color="#d1495b", zorder=3)
    ax.set(xlabel="Classification threshold", ylabel="F1 score", title="F1 score by threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f1_by_threshold.png", dpi=160)
    plt.close(fig)

    results = {
        "dataset_rows": int(len(df)),
        "split_sizes": {
            "train": int(len(df_train)),
            "validation": int(len(df_val)),
            "test": int(len(df_test)),
        },
        "categorical_features": categorical,
        "numerical_features": numerical,
        "q1_feature_auc": feature_aucs,
        "q1_answer": q1,
        "q2_validation_auc": float(validation_auc),
        "q3_precision_recall_closest_threshold": q3,
        "q3_precision": float(q3_row["precision"]),
        "q3_recall": float(q3_row["recall"]),
        "q3_absolute_gap": float(abs(q3_row["precision"] - q3_row["recall"])),
        "q4_best_f1_threshold": q4,
        "q4_best_f1": float(q4_row["f1"]),
        "q5_fold_auc_scores": [float(x) for x in cv_scores],
        "q5_fold_auc_std_population": cv_std,
        "q6_tuning": tuning,
        "q6_best_c": float(q6),
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print("ML Zoomcamp 2026 — Homework 4 results")
    print(f"Dataset: {len(df)} rows; split train/validation/test = {len(df_train)}/{len(df_val)}/{len(df_test)}")
    print("\nQ1 — ROC AUC feature importance (direction-adjusted):")
    for name, score in sorted(feature_aucs.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score:.6f}")
    print(f"  Answer: {q1}")
    print(f"\nQ2 — Validation ROC AUC: {validation_auc:.6f} (rounds to {validation_auc:.3f})")
    print(f"\nQ3 — Closest precision/recall threshold: {q3:.2f} (P={q3_row['precision']:.4f}, R={q3_row['recall']:.4f}, gap={abs(q3_row['precision'] - q3_row['recall']):.4f})")
    print(f"\nQ4 — Best F1 threshold: {q4:.2f} (F1={q4_row['f1']:.6f})")
    print(f"\nQ5 — Fold AUC scores: {[round(x, 6) for x in cv_scores]}")
    print(f"  Population standard deviation: {cv_std:.6f} (rounds to {cv_std:.3f})")
    print("\nQ6 — 5-fold results:")
    for c, values in tuning.items():
        print(f"  C={c:g}: mean AUC={values['mean_auc']:.6f}, std={values['std_auc']:.6f}")
    print(f"  Best C: {q6:g}")
    print(f"\nFigures saved to: {FIGURES_DIR}")
    return results


if __name__ == "__main__":
    main()
