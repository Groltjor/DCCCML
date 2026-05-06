from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def predict_with_threshold(
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert predicted probabilities into binary predictions using a threshold.
    """
    return (y_proba >= threshold).astype(int)


def evaluate_classifier(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a binary classifier using probability scores and a custom threshold.
    """
    y_pred = predict_with_threshold(y_proba, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def threshold_report(
    y_true,
    y_proba,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate classifier performance across multiple thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    rows = []

    for threshold in thresholds:
        metrics = evaluate_classifier(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold,
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def add_business_utility(
    threshold_df: pd.DataFrame,
    tn_value: float = 100,
    tp_value: float = 0,
    fp_cost: float = -100,
    fn_cost: float = -500,
) -> pd.DataFrame:
    """
    Add a simple business utility score to a threshold report.

    Default assumptions:
    - TN: correctly approved non-default customer
    - FP: customer incorrectly blocked
    - FN: risky customer incorrectly approved
    - TP: risky customer correctly blocked

    These values are illustrative and should be adapted to the business context.
    """
    df = threshold_df.copy()

    df["business_utility"] = (
        df["tn"] * tn_value
        + df["tp"] * tp_value
        + df["fp"] * fp_cost
        + df["fn"] * fn_cost
    )

    return df.sort_values("business_utility", ascending=False)

def compare_models(
    models: dict,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare multiple fitted models using the same test data and threshold.
    """
    rows = []

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_classifier(
            y_true=y_test,
            y_proba=y_proba,
            threshold=threshold,
        )

        metrics["model"] = model_name
        rows.append(metrics)

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )
