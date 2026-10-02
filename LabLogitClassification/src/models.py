import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from patsy import dmatrices
import statsmodels.discrete.discrete_model as sm
import statsmodels.formula.api as smf
import statsmodels.api as sma
from statsmodels.graphics.regressionplots import *
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import preprocessing



# Logistic classifiier based on sm
class LogClassifier():
    def __init__(self):
        raise NotImplementedError

    def train_model(self,y,X):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def predict_proba(self,X):
        raise NotImplementedError

    def predict(self,X,threshold = 0.5):
        raise NotImplementedError



#Now, lets use scikit-learn
class Knn_classifier(KNN):
    def __init__(self, n_neighbors = 5):
        raise NotImplementedError

    def train_model(self,y,X):
        raise NotImplementedError
