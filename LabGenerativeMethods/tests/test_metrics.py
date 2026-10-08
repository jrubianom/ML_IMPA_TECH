import numpy as np
import pytest

from src.metrics import accuracy, confusion_matrix, rates, roc_curve, auc


def test_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    assert np.array_equal(confusion_matrix(y_true, y_pred), [[2, 1], [1, 1]])
    assert accuracy(y_true, y_pred) == pytest.approx(0.6)


def test_rates():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    r = rates(y_true, y_pred)
    assert r["fpr"] == pytest.approx(1 / 3)
    assert r["tpr"] == pytest.approx(1 / 2)
    assert r["error"] == pytest.approx(2 / 5)


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr = roc_curve(y, perfect)
    assert (fpr[0], tpr[0]) == (0, 0) and (fpr[-1], tpr[-1]) == (1, 1)
    assert auc(fpr, tpr) == pytest.approx(1.0)
    assert auc(*roc_curve(y, 1 - perfect)) == pytest.approx(0.0)
