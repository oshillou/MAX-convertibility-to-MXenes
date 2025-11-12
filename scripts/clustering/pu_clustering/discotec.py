import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def compute_consensus_matrix(partitions):
    n_samples = partitions.shape[1]
    x, y = jnp.meshgrid(jnp.arange(n_samples), jnp.arange(n_samples))
    zz = jnp.c_[x.ravel(), y.ravel()]

    def compress(i,j):
        return (partitions[:,i]==partitions[:,j]).sum()

    func = jax.vmap(compress)
    return func(zz[:,0], zz[:,1]).reshape((n_samples, n_samples))/len(partitions)


@jax.jit
@partial(jax.vmap, in_axes=[0,None])
def compute_tv_ranking(y_pred, centroid):
    total = jax.lax.fori_loop(
        lower=0,
        upper=len(y_pred),
        body_fun=lambda i, i_value: i_value+jax.lax.fori_loop(lower=0, upper=len(y_pred), body_fun=lambda j, j_value: j_value+jnp.abs((y_pred[i]==y_pred[j])-centroid[i,j]), init_val=0), 
        init_val=0)
    return total/len(centroid)/len(centroid)


@partial(jax.vmap, in_axes=[0,None, None])
def compute_tpr_fpr(y_pred, centroid, threshold):
    confusion_matrix=jax.lax.fori_loop(
        lower=0,
        upper=len(y_pred),
        body_fun = lambda i, val_outer: jax.lax.fori_loop(
            lower=0,
            upper=len(y_pred),
            body_fun = lambda j, val_inner: val_inner.at[(y_pred[i]==y_pred[j]).astype(int), (centroid[i,j]>=threshold).astype(int)].add(1),
            init_val = val_outer,
        ),
        init_val = jnp.zeros((2,2))
    )
    return confusion_matrix[0,1]/confusion_matrix[0].sum(), confusion_matrix[1,1] / confusion_matrix[1].sum()

@jax.jit
def auc_discotec(y_pred):
    T = len(y_pred)
    centroid = compute_consensus_matrix(y_pred)
    unique_thresholds = jnp.unique(centroid, size=T+1, fill_value=jnp.inf)
    sorted_thresholds = jnp.sort(unique_thresholds, descending=True)

    ratio_function = jax.vmap(compute_tpr_fpr, in_axes=[None, None, 0])

    fpr, tpr = ratio_function(y_pred, centroid, sorted_thresholds)

    delta_fpr = jnp.diff(fpr, axis=0)
    trapezoid_tpr = tpr[1:]+tpr[:-1]

    return (delta_fpr*trapezoid_tpr).sum(0)/2

    
