from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    recall_score,
    f1_score,
    precision_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score
)

import matplotlib.pyplot as plt


def score_modelo_lineal(y_true : list[float], y_pred : list[float])-> pd.DataFrame:

    """
    Esta es una función para evaluar un modelo lineal
    """

    MSE = mean_squared_error(y_true, y_pred)
    RMSE = root_mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)


    return


def score_modelo_no_lineal(y_true : list[float], y_pred: list[float], y_scores: list[float]) -> tuple[Any]:
    """
    Esta es una función para evaluar el modelo de tipo no lineal
    """

    CM = confusion_matrix(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    ACCURACY = accuracy_score(y_true, y_pred)
    ROCAUC = roc_auc_score(y_true, y_scores) 
    RECALL = recall_score(y_true, y_pred)
    PRECISION = precision_score(y_true, y_pred, zero_division= 0)
    PR_AUC = average_precision_score(y_true, y_scores)

    return CM, F1, ACCURACY, ROCAUC, RECALL, PRECISION, PR_AUC


def plot_confusion_matrix(confusion_array, axes):

    disp = ConfusionMatrixDisplay(confusion_array)
    disp.plot(ax = axes)
    return disp

def plot_roc_curve(y_true, y_scores, axes):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    display = RocCurveDisplay(
        fpr = fpr,
        tpr = tpr,
        roc_auc = roc_auc,
    )

    display.plot(ax =  axes)
    axes.plot([0, 1], [0, 1], linestyle="--")

    return display


def plot_precision_recall_curve(y_true, y_score, axes):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_score
    )

    display_prec_recall = PrecisionRecallDisplay(
        precision,
        recall
    )

    display_prec_recall.plot(ax = axes)

    return display_prec_recall

def frame_data(F1, ACCURACY, PRECISION, ROCAUC, RECALL, PR_AUC):

    data = {
        'F1' : F1,
        'ACCURACY' : ACCURACY,
        'PRECISION' : PRECISION,
        'ROCAUC' : ROCAUC,
        'RECALL' : RECALL,
        'PR_AUC' : PR_AUC,
    }

    frame = pd.DataFrame(data, index = [0])
    
    return frame

