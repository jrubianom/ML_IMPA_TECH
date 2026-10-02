import pandas as pd
from patsy import dmatrices
import statsmodels.api as sma
import numpy as np

def load_data(path):
    raise NotImplementedError

def select_features(dataset,features,response,label):
    raise NotImplementedError

def select_features_split_data(train_dataset,test_dataset,
                                     features,response,label):
    raise NotImplementedError
