import numpy as np

from sklearn.cluster import KMeans
from metric_learn import RCA
from sklearn.utils.validation import check_random_state, check_X_y


class RCAKMeans:
    """
    Relative Components Analysis (RCA) + KMeans
    """

    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):

        X,y = check_X_y(X, y, ensure_2d=True, ensure_min_samples=self.n_clusters)
        random_state = check_random_state(self.random_state)
        
        # All positive samples are connected (unique fully connected components)
        # Other samples are not part of a connected component
        # 0: connected component, -1: no component
        chunks = (y-1).astype(int)

        rca = RCA()
        rca.fit(X, chunks=chunks)
        X_transformed = rca.transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=random_state)
        kmeans.fit(X_transformed)

        self.labels_ = kmeans.labels_

        return self