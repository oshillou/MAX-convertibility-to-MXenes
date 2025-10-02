import argparse
import pickle
import os
from pu_clustering import PUMKLKmeans
import pandas as pd
from sklearn import preprocessing
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--clustering_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="distances.csv")

    args = parser.parse_args()

    assert os.path.exists(args.model_file)
    assert os.path.exists(args.clustering_file)

    return args


def main():
    args = get_args()

    print("Loading model")
    with open(args.model_file, "rb") as file:
        model = pickle.load(file)

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0).to_numpy()
    X_cont = preprocessing.MinMaxScaler().fit_transform(X_cont)
    X_cat = pd.read_csv(args.categorical_data, index_col=0).to_numpy()
    X = np.concatenate([X_cont, X_cat], axis=1)

    print("Loading clusters")
    clusterings = pd.read_csv(args.clustering_file, index_col=0)

    print("Computing distances")
    kernels = [model._compute_kernel(X, **params) for params in model.kernel_params]
    kernel_matrix = sum([kernels[j]*model.kernel_weights_[j] for j in range(model.n_kernels_)])
    kernel_matrix = np.copy(kernel_matrix)


    largest_positive_cluster, = np.where(clusterings["Cluster"].to_numpy().reshape(-1)==1)

    # Distance between a centroid and a sample using a kernel matrix
    distances = np.copy(np.diag(kernel_matrix)) # Sample term
    distances += kernel_matrix[largest_positive_cluster][:,largest_positive_cluster].sum() # Centroid term
    distances -= 2 * kernel_matrix[:,largest_positive_cluster].sum(axis=1) # Adversarial term

    # Correction for floating point errors
    distances = np.maximum(distances, 0)

    print("Exporting distances")
    pd.Series(distances, name="Distance", index=clusterings.index).to_csv(args.output_file)
    

if __name__ == "__main__":
    main()