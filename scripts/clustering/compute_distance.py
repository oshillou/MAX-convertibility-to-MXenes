import argparse
import os
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--clustering_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="distances.csv")

    args = parser.parse_args()

    assert os.path.exists(args.clustering_file)

    return args


def main():
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0).to_numpy()
    X_cont = preprocessing.MinMaxScaler().fit_transform(X_cont)
    X_cat = pd.read_csv(args.categorical_data, index_col=0).to_numpy()
    X = np.concatenate([X_cont, X_cat], axis=1)

    print("Loading clusters")
    clusterings = pd.read_csv(args.clustering_file, index_col=0)
    y = clusterings["Cluster"].to_numpy().reshape((-1))

    print("Computing distances to centroid")
    centroid_big_cluster = X[y==1].mean(axis=0, keepdims=True)
    print(X.shape, centroid_big_cluster.shape)
    distances = metrics.pairwise_distances(X, centroid_big_cluster).reshape((-1))

    print("Exporting distances")
    pd.Series(distances, name="Distance", index=clusterings.index).to_csv(args.output_file)
    

if __name__ == "__main__":
    main()