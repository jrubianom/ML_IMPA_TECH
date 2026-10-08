import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from src import data, metrics, models, plots

path = "datasets/Smarket.csv"
features = ["Lag1", "Lag2"]

# Load data and split by time: train on 2001-2004, test on 2005
# TODO

# Fit our LDA, QDA and Naive Bayes, and the scikit-learn ones
# TODO

# Test accuracy, confusion matrix and ROC curve of each model; compare with scikit-learn
# TODO

# Save the metrics to figures/metrics.csv
# TODO

# Plot the decision boundaries (training data) and the ROC curves (test data), save to figures/
# TODO
