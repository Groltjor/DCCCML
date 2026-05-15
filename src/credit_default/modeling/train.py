from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "credit_default_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "credit_default_cache"),
)

import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_default.evaluation import (
    frame_data,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    score_modelo_no_lineal,
)
from credit_default.modeling.pipelines import (
    build_logistic_regression_pipeline_np1log_optimized,
)
from credit_default.utils import load_checkpoint


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vif_checkpoint"
MODELS_ONE_CLICK_DIR = PROJECT_ROOT / "models_one_click"

MODEL_PATH = MODELS_ONE_CLICK_DIR / "modelo_final.joblib"
THRESHOLD_PATH = MODELS_ONE_CLICK_DIR / "threshold.json"
METRICS_PATH = MODELS_ONE_CLICK_DIR / "final_model_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_ONE_CLICK_DIR / "threshold_test_results_df.csv"
FIGURE_PATH = MODELS_ONE_CLICK_DIR / "modelo_optimizado.webp"
SHAP_BAR_PATH = MODELS_ONE_CLICK_DIR / "shap_bar.webp"
SHAP_BEESWARM_PATH = MODELS_ONE_CLICK_DIR / "shap_beeswarm.webp"

C_OPTIMIZED = 0.1
SHAP_SAMPLE_SIZE = 1000

THRESHOLDS_TEST = np.round(np.arange(0.1, 0.9, 0.05), 2)

BENEFICIO_TP = 5000
BENEFICIO_TN = 1000
COSTO_FP = 1000
COSTO_FN = 10000


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_threshold_test_results_df(
    y_test: pd.Series,
    y_scores: np.ndarray,
) -> pd.DataFrame:
    results = []

    for threshold in THRESHOLDS_TEST:
        y_pred_test = (y_scores >= threshold).astype(int)

        CM, F1, ACCURACY, ROCAUC, RECALL, PRECISION, PR_AUC = score_modelo_no_lineal(
            y_test,
            y_pred_test,
            y_scores,
        )

        tn, fp, fn, tp = CM.ravel()

        results.append(
            {
                "THRESHOLD": threshold,
                "F1": F1,
                "ACCURACY": ACCURACY,
                "PRECISION": PRECISION,
                "ROCAUC": ROCAUC,
                "RECALL": RECALL,
                "PR_AUC": PR_AUC,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    threshold_test_results_df = pd.DataFrame(results)
    threshold_test_results_df["profit"] = (
        threshold_test_results_df["tp"] * BENEFICIO_TP
        + threshold_test_results_df["tn"] * BENEFICIO_TN
        - threshold_test_results_df["fp"] * COSTO_FP
        - threshold_test_results_df["fn"] * COSTO_FN
    )

    return threshold_test_results_df.sort_values("profit", ascending=False)


def save_optimized_model_figure(
    CM: np.ndarray,
    y_test: pd.Series,
    y_scores: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 7))

    plot_confusion_matrix(CM, axes=axes[0])
    plot_roc_curve(y_test, y_scores, axes=axes[1])
    plot_precision_recall_curve(y_test, y_scores, axes[2])

    figure.suptitle("Modelo Optimizado")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, format="webp", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _to_transformed_frame(
    transformed_data,
    feature_names: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    if sparse.issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    return pd.DataFrame(
        transformed_data,
        columns=feature_names,
        index=index,
    )


def save_shap_figures(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed = _to_transformed_frame(
        preprocessor.transform(X_train),
        feature_names,
        X_train.index,
    )

    X_test_sample = X_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=42,
    )
    X_test_transformed = _to_transformed_frame(
        preprocessor.transform(X_test_sample),
        feature_names,
        X_test_sample.index,
    )

    explainer = shap.LinearExplainer(logistic_model, X_train_transformed)
    shap_values = explainer(X_test_transformed)

    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, format="webp", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, format="webp", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    """
    Train and export the notebook 05 winner as a one-click artifact.
    """
    MODELS_ONE_CLICK_DIR.mkdir(parents=True, exist_ok=True)

    arreglo_frames, metadata_json = load_checkpoint(CHECKPOINT_PATH)
    X_train, X_test, y_train, y_test = arreglo_frames

    model = build_logistic_regression_pipeline_np1log_optimized(
        cols_to_log1p=metadata_json["columnas_log1p"],
        cols_to_numeric=metadata_json["columnas_numericas"],
        cols_to_yeo=metadata_json["columnas_yeo"],
        categorical_cols=metadata_json["columnas_categoricas"],
        C_optimized=C_OPTIMIZED,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    CM, F1, ACCURACY, ROCAUC, RECALL, PRECISION, PR_AUC = score_modelo_no_lineal(
        y_test,
        y_pred,
        y_scores,
    )

    scores_frame = frame_data(
        F1,
        ACCURACY,
        PRECISION,
        ROCAUC,
        RECALL,
        PR_AUC,
    )

    threshold_test_results_df = build_threshold_test_results_df(y_test, y_scores)

    joblib.dump(model, MODEL_PATH)
    threshold_test_results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)
    save_optimized_model_figure(CM, y_test, y_scores)
    save_shap_figures(model, X_train, X_test)

    best_threshold_row = threshold_test_results_df.iloc[0].to_dict()
    for count_col in ["tn", "fp", "fn", "tp", "profit"]:
        best_threshold_row[count_col] = int(best_threshold_row[count_col])

    with THRESHOLD_PATH.open("w") as f:
        json.dump({"threshold": best_threshold_row["THRESHOLD"]}, f, indent=2)

    with METRICS_PATH.open("w") as f:
        json.dump(
            {
                "default_threshold_metrics": {
                    key: _json_safe(value)
                    for key, value in scores_frame.iloc[0].to_dict().items()
                },
                "best_profit_threshold": {
                    key: _json_safe(value) for key, value in best_threshold_row.items()
                },
                "model": "build_logistic_regression_pipeline_np1log_optimized",
                "C_optimized": C_OPTIMIZED,
            },
            f,
            indent=2,
        )

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Threshold results saved to: {THRESHOLD_RESULTS_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")
    print(f"SHAP bar saved to: {SHAP_BAR_PATH}")
    print(f"SHAP beeswarm saved to: {SHAP_BEESWARM_PATH}")


if __name__ == "__main__":
    main()
