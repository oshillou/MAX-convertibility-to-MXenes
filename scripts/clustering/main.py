
from pu_clustering import PUKernelKMeans, PUMKLKmeans, COPKMeans, PCKMeans, MKMeans
import argparse
from sklearn import preprocessing, cluster, metrics
import pandas as pd
import numpy as np
import os
import itertools
from active_semi_clustering.semi_supervised import pairwise_constraints
import pickle

def verbose_print(config, *args, **kwargs):
    if config.verbose:
        print(*args, **kwargs)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--clustering_model", type=str, default="pumkl", choices=["pumkl", "kmeans", "pckmeans", "copkmeans", "mkmeans", "mpckmeans"])
    parser.add_argument("--model_file", type=str, default="model.pkl")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--distance_file", type=str, default="distances.csv")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--validate_max", type=int, default=-1)
    parser.add_argument("--validate_mxene", type=int, default=-1)
    parser.add_argument("--verbose", default=False, action="store_true")

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)
    assert os.path.exists(args.pu_labels)

    return args


def main():
    args = get_args()

    verbose_print(args, "Reading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0).to_numpy()
    X_cont = preprocessing.MinMaxScaler().fit_transform(X_cont)
    X_cat = pd.read_csv(args.categorical_data, index_col=0).to_numpy()
    X = np.concatenate([X_cont, X_cat], axis=1)
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)

    validation_indices = np.zeros(len(y))

    if args.validate_max!=-1:
        assert args.validate_max < (y>0).sum()
        positive_samples, = np.nonzero(y)
        validation_indices[positive_samples[args.validate_max]] = 1
    elif args.validate_mxene!=-1:
        assert args.validate_mxene < y.max()
        validation_indices[y==args.validate_mxene+1] = 1
    
    # Finally, before clustering, convert back targets to simple binary targets
    # indicating whether samples should be together or we do not know
    y[validation_indices==1] = 0
    y = (y>0).astype(int)

    X_train, y_train = X, y


    verbose_print(args, "Performing clustering")
    if args.clustering_model == "pumkl":
        kernels_params = [
            {"metric":"linear"},
            {"metric":"rbf"},
            {"metric":"laplacian"},
            {"metric":"additive_chi2"}
        ]
        max_kernels = 4
    
        base_model = PUKernelKMeans(n_clusters=args.n_clusters, tol=20/len(X_train))
        model = PUMKLKmeans(base_model, kernel_params=kernels_params, n_candidates=200, n_initial_candidates=50, max_kernels=max_kernels, verbose=args.verbose, random_state=args.seed)
    elif args.clustering_model == "kmeans":
        model = cluster.KMeans(n_clusters=args.n_clusters, random_state=args.seed)
    else:
        ml_constraints = list(itertools.combinations(np.nonzero(y)[0], r=2))
        if args.clustering_model == "pckmeans":
            if y_train.sum() > 1: # More than one constrained sample
                D = metrics.pairwise_distances(X_train[y_train==1], X_train, metric="sqeuclidean")
                std_dist = D[np.triu_indices(len(D), k=1)].std()
                del D
                w = 2*std_dist / y_train.sum()
            else:
                w = np.ones(1) # Weight for a single constraint
            model = PCKMeans(n_clusters=args.n_clusters, w=w, random_state=args.seed)
        elif args.clustering_model == "copkmeans":
            model = COPKMeans(n_clusters=args.n_clusters, random_state=args.seed)
        elif args.clustering_model == "mpckmeans":
            model = pairwise_constraints.MPCKMeans(n_clusters=args.n_clusters)
        elif args.clustering_model == "mkmeans":
            model = MKMeans(n_clusters=args.n_clusters)



    verbose_print(args, "Fitting model")
    if args.clustering_model != "mpckmeans":
        model.fit(X_train,y_train)
    else:
        model.fit(X_train, ml=ml_constraints)

    verbose_print(args, f"Exporting model to {args.model_file}")
    with open(args.model_file, "wb") as file:
        pickle.dump(model, file)


    verbose_print(args, "Converting clusters to positive/negative clusters")
    clustering = model.labels_
    positive_clusters = np.unique(clustering[y==1])
    final_clusters = np.zeros(len(X), dtype=int)
    for i, k in enumerate(positive_clusters):
        final_clusters[clustering==k] = i+1
    
    pd.Series(final_clusters, name="Cluster", index=pu_labels.index).to_csv(args.clustering_file)

    verbose_print(args,"Computing the distances between samples and the largest clusters")
    # Note that by convention in the different clustering algorithms, the largest cluster
    # is the number 1
    if args.clustering_model == "pumkl":
        # We have to use a kernel distance
        kernels = [model._compute_kernel(X, **params) for params in model.kernel_params]
        kernel_matrix = sum([kernels[j] * model.kernel_weights_[j] for j in range(model.n_kernels_)])
        kernel_matrix = np.copy(kernel_matrix)
        largest_positive_cluster, = np.where(clustering == 1)

        # Distance between a centroid and a sample using a kernel matrix
        distances = np.copy(np.diag(kernel_matrix))  # Sample term
        distances += kernel_matrix[largest_positive_cluster][:, largest_positive_cluster].sum()  # Centroid term
        distances -= 2 * kernel_matrix[:, largest_positive_cluster].sum(axis=1)  # Adversarial term

        distances = np.maximum(distances, 0)
    else:
        # The regular Euclidean distance
        centroid_largest_cluster = X[clustering==1].mean(axis=0, keepdims=True)
        distances = metrics.pairwise_distances(X, centroid_largest_cluster).reshape((-1,))

    verbose_print(args, "Exporting distances")
    pd.Series(distances, name="Distance", index=pu_labels.index).to_csv(args.distance_file)



if __name__ == "__main__":
    main()