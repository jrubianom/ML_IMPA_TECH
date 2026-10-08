import numpy as np


def log_gaussian(X, mu, Sigma):
    """log N(x | mu, Sigma) for every row of X.

    Hint: np.linalg.inv, np.linalg.slogdet.
    """
    raise NotImplementedError


def softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)


class GenerativeClassifier:
    """Base class: fit P(Y=k) and P(X|Y=k), classify with Bayes' rule.

    Subclasses implement `_fit_class_conditionals` and `_log_likelihoods`.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)
        # TODO: self.priors_ (shape (K,)) and self.means_ (shape (K, p))
        raise NotImplementedError
        self._fit_class_conditionals(X, y, n)
        return self

    def discriminant(self, X):
        """delta_k(x) = log pi_k + log f_k(x), shape (n, K)."""
        X = np.asarray(X, dtype=float)
        return np.log(self.priors_) + self._log_likelihoods(X)

    def predict_proba(self, X):
        return softmax(self.discriminant(X))

    def predict(self, X, threshold=None):
        """Most probable class. With two classes, `threshold` on P(Y = classes_[1] | x) can be given."""
        raise NotImplementedError


class LDAClassifier(GenerativeClassifier):
    """X | Y=k ~ N(mu_k, Sigma), with Sigma shared by all classes."""

    def _fit_class_conditionals(self, X, y, n):
        # TODO: pooled covariance divided by n - K, stored in self.covariance_
        raise NotImplementedError

    def _log_likelihoods(self, X):
        raise NotImplementedError

    def linear_coefficients(self):
        """delta_k(x) = a_k + b_k^T x. Returns (a, b) with shapes (K,), (K, p)."""
        raise NotImplementedError


class QDAClassifier(GenerativeClassifier):
    """X | Y=k ~ N(mu_k, Sigma_k), one covariance per class."""

    def _fit_class_conditionals(self, X, y, n):
        # TODO: self.covariance_ with shape (K, p, p), each divided by n_k - 1
        raise NotImplementedError

    def _log_likelihoods(self, X):
        raise NotImplementedError


class GaussianNBClassifier(GenerativeClassifier):
    """Features independent given the class: X_j | Y=k ~ N(mu_kj, sigma_kj^2)."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def _fit_class_conditionals(self, X, y, n):
        # TODO: self.var_ with shape (K, p), variances divided by n_k,
        # plus var_smoothing * (largest variance of all features), as scikit-learn does
        raise NotImplementedError

    def _log_likelihoods(self, X):
        raise NotImplementedError
