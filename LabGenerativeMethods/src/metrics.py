import numpy as np
import pandas as pd


def accuracy(y_true, y_pred):
    raise NotImplementedError


def confusion_matrix(y_true, y_pred):
    """Rows: true class (0, 1). Columns: predicted class (0, 1)."""
    raise NotImplementedError


def rates(y_true, y_pred):
    """Dict with keys 'error', 'fpr', 'tpr', 'precision'."""
    raise NotImplementedError


def roc_curve(y_true, scores):
    """FPR and TPR for every threshold, ordered from (0, 0) to (1, 1)."""
    raise NotImplementedError


def auc(fpr, tpr):
    raise NotImplementedError


def evaluate_model(y_true, y_pred):
    return {"accuracy": accuracy(y_true, y_pred), **rates(y_true, y_pred)}


def save_metrics(metrics, path):
    pd.DataFrame(metrics).T.to_csv(path)
