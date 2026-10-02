import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def accuracy(y_true, y_pred):
    raise NotImplementedError

def precision(y_true, y_pred):
    raise NotImplementedError

def evaluate_model(y_true, y_pred):
    raise NotImplementedError

def save_metrics(metrics, path):
    raise NotImplementedError
