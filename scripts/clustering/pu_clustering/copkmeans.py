# This code was copied and adapted from:
# https://github.com/datamole-ai/active-semi-supervised-clustering/blob/master/active_semi_clustering/semi_supervised/pairwise_constraints/copkmeans.py

# Changes include: removal of ML/CL constraints to only consider ML constraints (PU setting)
# Addition of a deterministic random state

import numpy as np
from sklearn.utils.validation import check_random_state, check_X_y
from sklearn import metrics

class COPKMeans:
    def __init__(self, n_clusters=3, max_iter=100, max_retries_per_iter= 1000, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_retries_per_iter = max_retries_per_iter

    def fit(self, X, y=None):

        X,y = check_X_y(X, y, ensure_2d=True, ensure_min_samples=self.n_clusters)
        random_state = check_random_state(self.random_state)

        positive_indices, = np.nonzero(y)
        ml_graph = {i: set() for i in range(len(X))}
        for i in positive_indices:
            ml_graph[i] = set(np.setxor1d([i], positive_indices))

        # Initialize cluster centers (exclude positive samples) to medoids
        # Thus, there cannot be empty clusters: only one medoid will be shifted 
        negative_indices = np.setxor1d(np.arange(len(X)), positive_indices)
        cluster_centers = X[random_state.choice(negative_indices, self.n_clusters, replace=False), :]

        # Repeat until convergence
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self._assign_clusters(X, random_state, cluster_centers, ml_graph)

            # Estimate means
            cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged:
                break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self
    
    def predict(self, X):
        return metrics.pairwise_distances(X, self.cluster_centers_, metric="sqeuclidean").argmin(axis=1)

    def _assign_clusters(self, *args):
        for _ in range(self.max_retries_per_iter):
            try:
                return self._try_assign_clusters(*args)
            except:
                continue

        raise Exception("Deadlock")

    def _try_assign_clusters(self, X, random_state, cluster_centers, ml_graph):
        labels = -np.ones(len(X), dtype=int)

        data_indices = list(range(X.shape[0]))
        random_state.shuffle(data_indices)

        distances = metrics.pairwise_distances(X, cluster_centers, metric="sqeuclidean")
        
        for i in data_indices:
            local_distances = distances[i]
            # sorted_cluster_indices = np.argsort([dist(x, c) for c in cluster_centers])

            for cluster_index in local_distances.argsort():
                if not self._violates_constraints(i, cluster_index, labels, ml_graph):
                    labels[i] = cluster_index
                    break
            # Since we do not have cannot-link constraints, all samples are necessarily assigned to a cluster

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise Exception("Empty cluster found")
        return labels

    def _violates_constraints(self, i, cluster_index, labels, ml_graph):
        for j in ml_graph[i]:
            if labels[j] > 0 and cluster_index != labels[j]:
                return True
        return False