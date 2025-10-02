import jax
from jax import numpy as jnp
from functools import partial

# For parameter validation
from sklearn import base
from numbers import Integral, Real
from sklearn.utils._param_validation import Interval



class PUKernelKMeans(base.BaseEstimator, base.ClusterMixin):
    _parameter_constraints = {
        "n_clusters":[Interval(Integral, 1, None, closed="left")],
        "n_iterations":[Interval(Integral, 1, None, closed="left")],
        "tol":[Interval(Real, 0, None, closed="left")]
    }
    def __init__(self, n_clusters=3, n_iterations=1000, tol=1e-3):
        self.n_clusters = n_clusters
        self.tol = tol
        self.n_iterations = n_iterations
        
    
    def fit(self, kernel, pu_labels):

        # Perform data validation (function call only valid for scikit learn <1.6)
        kernel, pu_labels = self._validate_data(kernel, pu_labels, accept_sparse=False, ensure_2d=True, ensure_min_samples=self.n_clusters)

        initial_memberships = _pu_init(pu_labels, kernel, self.n_clusters)
        final_memberships = _kernelkmeans(self.n_iterations, self.tol, self.n_clusters, kernel, initial_memberships)

        self.labels_ = final_memberships
        return self
    
    def fit_predict(self, kernel, pu_labels):
        return self.fit(kernel, pu_labels).labels_
    
    def predict(self, training_kernel, new_samples_kernel):
        assert len(training_kernel) == training_kernel.shape[1]
        assert len(training_kernel) == len(self.labels_)
        assert new_samples_kernel.shape[1] == len(self.labels_)

        cluster_indicator = jnp.eye(self.n_clusters)[self.labels_]
        cluster_sizes = jnp.bincount(self.labels_)

        centroid_product = jnp.diag(cluster_indicator.T@training_kernel@cluster_indicator)/jnp.square(cluster_sizes) # Shape K
        new_samples_product = new_samples_kernel@cluster_indicator / cluster_sizes

        distances = -2*new_samples_product+centroid_product.reshape((1,-1))
        # Since we take the argmin over \mu_k, there is no need to compute <phi(x), phi(x)>

        return distances.argmin(axis=1)


@partial(jax.jit, static_argnums=[2])
def _distance2clusters(cluster_memberships, kernel_matrix, n_clusters):

    cluster_sizes = jnp.bincount(cluster_memberships, length=n_clusters)

    @partial(jax.vmap, in_axes=[None, 0])
    @partial(jax.vmap, in_axes=[0, None])
    def sum_cluster_values(k, kernel_entries):
        indicator = cluster_memberships==k
        return (kernel_entries*indicator).sum()

    kernel_value_per_sample = sum_cluster_values(jnp.arange(n_clusters), kernel_matrix) # Shape n x K

    cluster_indicator = jnp.eye(n_clusters)[cluster_memberships]
    kernel_value_per_cluster = (kernel_value_per_sample * cluster_indicator).sum(axis=0)

    distance_matrix = (kernel_value_per_cluster/jnp.square(cluster_sizes)).reshape((1,-1))-2*kernel_value_per_sample/cluster_sizes

    return distance_matrix

@partial(jax.jit, static_argnums=[2])
def _pu_init(pu_labels, kernel_matrix, n_clusters):

    distances = jnp.zeros((len(pu_labels), n_clusters))
    
    # We begin by computing the distance between all samples and cluster 1
    distance_to_pu_labels = _distance2clusters(pu_labels, kernel_matrix, 2) # Shape nx2
    # The function above computes only k(mu, mu) -2*k(x, mu)
    # So we must add to each entry k(x,x)
    distances = distances.at[:,0].set(distance_to_pu_labels[:,1]+jnp.diag(kernel_matrix))

    @jax.jit
    def add_new_centroid(k, carry):
        distances, is_centroid = carry
        # Compute the average distance to all clusters
        avg_distance = distances.sum(axis=1)/k
        # Delete centroid entries
        avg_distance = avg_distance*(1-is_centroid)
        # Pick the highest entry
        selected_centroid = avg_distance.argmax()

        # Compute distances to new centroid (i.e. medoid)
        distance_to_centroid = jnp.diag(kernel_matrix)+kernel_matrix[selected_centroid, selected_centroid]-2*kernel_matrix[selected_centroid]

        return distances.at[:,k].set(distance_to_centroid), is_centroid.at[selected_centroid].set(1)

    # Now, for all remaining clusters, we must iteratively compute the farthest one (which is not a centroid)
    final_distances, final_centroids = jax.lax.fori_loop(lower=1, upper=n_clusters, body_fun=add_new_centroid, init_val=(distances, pu_labels))

    # And finally, compute the assignments
    final_memberships = jnp.where(pu_labels==1, 0, final_distances.argmin(axis=1))

    return final_memberships

@partial(jax.jit, static_argnums=[0,1,2])
def _kernelkmeans(n_iter, tol, n_clusters, kernel_matrix, cluster_memberships):

    @jax.jit
    def kernel_kmeans_update(carry):
        i, _, cluster_memberships, kernel_matrix = carry
        distance = _distance2clusters(cluster_memberships, kernel_matrix, n_clusters)
        return i+1, cluster_memberships, distance.argmin(axis=1), kernel_matrix

    @jax.jit
    def stopping_criterion(carry):
        i, old_memberships, cluster_memberships, _ = carry
        return (jnp.mean(old_memberships==cluster_memberships)<tol) & (i<n_iter) 

    init_value = (0, jnp.arange(len(cluster_memberships)), cluster_memberships, kernel_matrix)
    results = jax.lax.while_loop(stopping_criterion, kernel_kmeans_update, init_val=init_value)

    return results[2]
    