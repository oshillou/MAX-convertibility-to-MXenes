import numpy as np
from sklearn import metrics
from sklearn.utils.validation import check_random_state, check_X_y


class PCKMeans:
    def __init__(self, n_clusters=3, max_iter=100, w=1, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        self.random_state = random_state

    def fit(self, X, y=None):

        X,y = check_X_y(X, y, ensure_2d=True, ensure_min_samples=self.n_clusters)
        random_state = check_random_state(self.random_state)

        positive_indices, = np.nonzero(y)

        # Initialize centroids
        positive_center = X[positive_indices].mean(axis=0, keepdims=True)

        negative_indices = np.setxor1d(np.arange(len(X)), positive_indices)
        remaining_cluster_centers = X[random_state.choice(negative_indices, self.n_clusters - 1, replace=False), :]
        cluster_centers = np.concatenate([positive_center, remaining_cluster_centers], axis=0)

        # Repeat until convergence
        for _ in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, random_state)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged:
                break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _assign_clusters(self, X, y, cluster_centers, random_state):
        labels = -np.ones(len(X), dtype=int)

        index = np.arange(len(X))
        random_state.shuffle(index)

        distances = 0.5*metrics.pairwise_distances(X, cluster_centers, metric="sqeuclidean")

        positive_per_cluster = np.zeros(self.n_clusters) # Keep track of the number of positive samples per cluster

        for i in index:
            if y[i]==1:
                # Positive sample, so we have must-link constraints to ensure
                penalties_per_cluster = self.w*(positive_per_cluster.sum()-positive_per_cluster)
                selected_cluster = np.argmin(distances[i]+penalties_per_cluster)
                positive_per_cluster[selected_cluster] += 1
                labels[i] = selected_cluster
            else:
                # Unlabelled sample, we have not constraint to satisfy
                labels[i] = distances[i].argmin()

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise Exception("Empty clusters")

        return labels
    
    def predict(self, X):
        return metrics.pairwise_distances(X, self.cluster_centers_, metric="sqeuclidean").argmin(axis=1)
    
    