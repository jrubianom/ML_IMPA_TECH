import numpy as np
import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def select_features(dataset, features, response, label):
    """Return y (1 where response == label, else 0) and X (numpy array with the given features)."""
    raise NotImplementedError


def split_by_year(dataset, year):
    """Train: rows before `year`. Test: rows from `year` on."""
    raise NotImplementedError


def select_features_split_data(train_dataset, test_dataset, features, response, label):
    raise NotImplementedError
