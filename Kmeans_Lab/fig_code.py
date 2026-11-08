"""Helper for Clustering_Kmeans_v4.ipynb.

Port of plot_kmeans_interactive from jakevdp/sklearn_pycon2015 (fig_code),
updated for current scikit-learn and ipywidgets.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances


def plot_kmeans_interactive(min_clusters=1, max_clusters=6):
    X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=0.60)

    def _kmeans_step(frame=0, n_clusters=4):
        rng = np.random.RandomState(2)
        labels = np.zeros(X.shape[0])
        centers = rng.randn(n_clusters, 2)

        nsteps = frame // 3
        for i in range(nsteps + 1):
            old_centers = centers
            if i < nsteps or frame % 3 > 0:
                labels = euclidean_distances(X, centers).argmin(1)
            if i < nsteps or frame % 3 > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    centers = np.array([X[labels == j].mean(0)
                                        for j in range(n_clusters)])
                nans = np.isnan(centers)
                centers[nans] = old_centers[nans]

        plt.figure(figsize=(7, 5))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow',
                    vmin=0, vmax=n_clusters - 1)
        plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                    c=np.arange(n_clusters), s=200, cmap='rainbow')
        plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                    c='black', s=50)

        if frame % 3 == 2:
            for i in range(n_clusters):
                plt.annotate('', centers[i], old_centers[i],
                             arrowprops=dict(arrowstyle='->', linewidth=1))
            plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c=np.arange(n_clusters), s=200, cmap='rainbow')
            plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c='black', s=50)

        plt.xlim(-4, 4)
        plt.ylim(-2, 10)

        if frame % 3 == 1:
            plt.text(3.8, 9.5, "1. Reassign points to nearest centroid",
                     ha='right', va='top', size=14)
        elif frame % 3 == 2:
            plt.text(3.8, 9.5, "2. Update centroids to cluster means",
                     ha='right', va='top', size=14)
        plt.show()

    return interact(_kmeans_step,
                    frame=IntSlider(min=0, max=50, value=0),
                    n_clusters=IntSlider(min=min_clusters, max=max_clusters,
                                         value=4))
