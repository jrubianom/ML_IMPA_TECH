import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(ax, model, X, y, title="", n_grid=200, pad=0.5):
    """Posterior P(Y=1|x) as background, boundary P = 0.5 in black. 2D features only."""
    x1 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n_grid)
    x2 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n_grid)
    g1, g2 = np.meshgrid(x1, x2)
    prob = model.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 1].reshape(g1.shape)
    ax.contourf(g1, g2, prob, levels=np.linspace(0, 1, 11), cmap="bwr", alpha=0.3)
    ax.contour(g1, g2, prob, levels=[0.5], colors="k")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=12, alpha=0.6)
    ax.set(title=title)


def plot_roc(ax, curves):
    """curves: dict name -> (fpr, tpr, auc)."""
    for name, (fpr, tpr, a) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    ax.legend()
