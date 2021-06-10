import numpy as np
from sklearn.decomposition import PCA


def preproc_pca(dataset, n_features):
    """Dimensionality reduction technique: Principle Compoenent Analysis"""
    # Features for training
    X = np.array(dataset)[:, :-1]

    # Labels
    y = np.array(dataset)[:, -1]

    new_X = PCA(n_components=n_features).fit_transform(X)
    print(X.shape, new_X.shape)

    return new_X, y
