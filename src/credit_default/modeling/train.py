from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from credit_default.data import load_credit_data
from credit_default.features import (
    clean_credit_data,
    split_features_target,
    get_feature_groups,
)
from credit_default.modeling.pipelines import build_xgboost_pipeline
from credit_default.evaluation import (
    evaluate_classifier,
    threshold_report,
    add_business_utility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "default_of_credit_card_clients.xls"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

MODEL_PATH = MODELS_DIR / "xgboost_credit_default.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
METRICS_PATH = REPORTS_DIR / "final_model_metrics.json"
THRESHOLD_REPORT_PATH = REPORTS_DIR / "threshold_report.csv"


FINAL_THRESHOLD = 0.30


def main() -> None:
    """
    Train the final credit default classification model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_credit_data(DATA_PATH)
    clean_data = clean_credit_data(raw_data)

    X, y = split_features_target(clean_data)
    feature_groups = get_feature_groups()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_xgboost_pipeline(
        numeric_cols=feature_groups.numeric_cols,
        categorical_cols=feature_groups.categorical_cols,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    final_metrics = evaluate_classifier(
        y_true=y_test,
        y_proba=y_proba,
        threshold=FINAL_THRESHOLD,
    )

    thresholds_df = threshold_report(
        y_true=y_test,
        y_proba=y_proba,
    )

    utility_df = add_business_utility(thresholds_df)

    joblib.dump(model, MODEL_PATH)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": FINAL_THRESHOLD}, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=2)

    utility_df.to_csv(THRESHOLD_REPORT_PATH, index=False)

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Threshold report saved to: {THRESHOLD_REPORT_PATH}")


if __name__ == "__main__":
    main()