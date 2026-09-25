import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Tuple


def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    return float(eer)


def calculate_min_tdcf(
        scores: np.ndarray,
        labels: np.ndarray,
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0
) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    p_target_norm = p_target
    p_nontarget_norm = 1 - p_target

    tdcf = c_miss * fnr * p_target_norm + c_fa * fpr * p_nontarget_norm

    min_tdcf = np.min(tdcf)

    return float(min_tdcf)


def calculate_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return float(roc_auc)