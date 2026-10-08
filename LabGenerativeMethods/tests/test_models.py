import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from src.models import LDAClassifier, QDAClassifier, GaussianNBClassifier


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X0 = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 300)
    X1 = rng.multivariate_normal([1.5, 1], [[2, -0.3], [-0.3, 0.5]], 200)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(300, int), np.ones(200, int)]
    return X, y


def test_priors_and_means(data):
    X, y = data
    model = LDAClassifier().fit(X, y)
    assert np.allclose(model.priors_, [0.6, 0.4])
    assert np.allclose(model.means_[1], X[y == 1].mean(axis=0))


def test_lda_pooled_covariance(data):
    X, y = data
    model = LDAClassifier().fit(X, y)
    n0, n1 = 300, 200
    S0, S1 = np.cov(X[y == 0], rowvar=False), np.cov(X[y == 1], rowvar=False)
    expected = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    assert np.allclose(model.covariance_, expected)


def test_lda_matches_sklearn(data):
    X, y = data
    ours = LDAClassifier().fit(X, y)
    ref = LinearDiscriminantAnalysis().fit(X, y)
    assert np.array_equal(ours.predict(X), ref.predict(X))
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X))


def test_lda_is_linear(data):
    X, y = data
    model = LDAClassifier().fit(X, y)
    a, b = model.linear_coefficients()
    delta = model.discriminant(X)
    # the quadratic term is common to both classes, so differences must match exactly
    assert np.allclose(delta[:, 1] - delta[:, 0], (a[1] - a[0]) + X @ (b[1] - b[0]))


def test_qda_matches_sklearn(data):
    X, y = data
    ours = QDAClassifier().fit(X, y)
    ref = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    assert np.allclose(ours.covariance_, ref.covariance_)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X))


def test_nb_matches_sklearn(data):
    X, y = data
    ours = GaussianNBClassifier().fit(X, y)
    ref = GaussianNB().fit(X, y)
    assert np.allclose(ours.var_, ref.var_)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X))


@pytest.mark.parametrize("Model", [LDAClassifier, QDAClassifier, GaussianNBClassifier])
def test_proba_and_threshold(data, Model):
    X, y = data
    model = Model().fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.all(model.predict(X, threshold=0.0) == 1)
    assert np.all(model.predict(X, threshold=1.0) == 0)
    assert np.array_equal(model.predict(X), model.predict(X, threshold=0.5))
