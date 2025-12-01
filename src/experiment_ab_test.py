"""
experiment_ab_test.py

Runs an A/B experiment:

Model A: numeric-only Logistic Regression
Model B: numeric + TF-IDF of msg_content

Implements:
- Results-driven Analytical Method (simplified in code comments)
- A/B comparison of metrics
- Statistical test (t-test) on cross-validated F1 scores
- Saves results to outputs/ab_test_results.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from model_train import (
    build_model_A,
    build_model_B,
    evaluate_model,
    cross_val_f1,
    RANDOM_STATE,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cleaned_path = project_root / "data" / "discord_cleaned.csv"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. "
            "Run: python src/data_prep.py first."
        )

    # -------------------------------
    # Results-driven Analytical Method (RDM)
    # -------------------------------
    # 1. Understand problem: reduce scam exposure, improve phishing detection.
    # 2. Start at the end: we want higher F1 (balanced precision/recall) for phishing label.
    # 3. Identify additional resources: numeric features + text content.
    # 4. Obtain / prepare data: use cleaned CSV from data_prep.py.
    # 5. Do the work: A/B comparison of two model families.
    # 6. Present a minimum viable answer: metrics + hypothesis test.
    # 7. Iterate if necessary: tweak features/hyperparameters (outside scope of this script).

    print(f"[INFO] Loading cleaned dataset: {cleaned_path}")
    df = pd.read_csv(cleaned_path)

    feature_cols_numeric = [
        "msg_timestamp",
        "usr_joined_at",
        "time_since_join",
        "message_length",
        "word_count",
        "has_link",
        "has_mention",
        "num_roles",
    ]
    X_all = df[feature_cols_numeric + ["msg_content"]]
    y_all = df["label"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )

    # -------------------------------
    # Model A: numeric-only
    # -------------------------------
    print("\n[INFO] Training Model A (numeric-only)...")
    model_A = build_model_A()
    metrics_A = evaluate_model(model_A, X_train, X_test, y_train, y_test)
    f1_scores_A = cross_val_f1(model_A, X_all[feature_cols_numeric], y_all, cv=5)

    # -------------------------------
    # Model B: numeric + text
    # -------------------------------
    print("\n[INFO] Training Model B (numeric + TF-IDF text)...")
    model_B = build_model_B()
    metrics_B = evaluate_model(model_B, X_train, X_test, y_train, y_test)
    f1_scores_B = cross_val_f1(model_B, X_all, y_all, cv=5)

    # -------------------------------
    # Hypothesis test (statistical comparison)
    # -------------------------------
    # H0: Model B does NOT significantly improve F1 compared to Model A.
    # H1: Model B significantly improves F1.

    t_stat, p_value = ttest_ind(f1_scores_A, f1_scores_B, equal_var=False)

    print("\n[INFO] Cross-validated F1 scores")
    print(f"Model A F1 scores: {np.round(f1_scores_A, 4)}")
    print(f"Model B F1 scores: {np.round(f1_scores_B, 4)}")
    print(f"\n[INFO] t-test results: t = {t_stat:.4f}, p = {p_value:.6f}")

    if p_value < 0.05:
        decision = "Reject H0 (Model B significantly improves F1)"
    else:
        decision = "Fail to reject H0 (No significant difference)"

    print(f"[INFO] Decision: {decision}")

    # -------------------------------
    # Save results
    # -------------------------------
    results = pd.DataFrame(
        [
            {
                "model": "Model_A_numeric_only",
                **metrics_A,
                "cv_f1_mean": f1_scores_A.mean(),
                "cv_f1_std": f1_scores_A.std(),
            },
            {
                "model": "Model_B_numeric_plus_text",
                **metrics_B,
                "cv_f1_mean": f1_scores_B.mean(),
                "cv_f1_std": f1_scores_B.std(),
            },
        ]
    )

    results_path = output_dir / "ab_test_results.csv"
    results.to_csv(results_path, index=False)

    print(f"\n[SUCCESS] A/B metrics saved to: {results_path}")
    print("[INFO] Summary:")
    print(results)
    print(f"\n[INFO] Hypothesis test decision: {decision}")


if __name__ == "__main__":
    main()
