import numpy as np

from sklearn.cluster import KMeans
from metric_learn import MMC
from sklearn.utils.validation import check_X_y, check_random_state
import itertools

class MKMeans:
    def __init__(self, n_clusters=3, max_iter=1000, diagonal=True, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.diagonal = diagonal
        self.random_state = random_state
        self.mmc = None
        self.kmeans = None

    def fit(self, X, y=None):

        X,y = check_X_y(X, y, ensure_2d=True, ensure_min_samples=self.n_clusters)
        random_state = check_random_state(self.random_state)

        X_transformed = X
        
        constraints = list(itertools.combinations(np.nonzero(y)[0], r=2))
        self.mmc = MMC(diagonal=self.diagonal, random_state=random_state)
        self.mmc.fit(X, constraints=constraints)
        X_transformed = self.mmc.transform(X)

        self.kmeans = KMeans(n_clusters=self.n_clusters, init='random', max_iter=self.max_iter)
        self.kmeans.fit(X_transformed)

        self.labels_ = self.kmeans.labels_

        return self
    
    def predict(self, X):
        X_transformed = self.mmc.transform(X)
        return self.kmeans.predict(X_transformed)
    
    