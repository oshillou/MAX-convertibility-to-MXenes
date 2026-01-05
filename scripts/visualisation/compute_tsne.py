import argparse
import os
import pandas as pd
import numpy as np
from sklearn import metrics, manifold, preprocessing
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--output_file", type=str, default="tsne.pdf")

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)
    assert os.path.exists(args.pu_labels)

    return args


def main():
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels > 0] = 1



    print("Computing and exporting TSNE")
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    Z = manifold.TSNE().fit_transform(X_scaled)
    plt.figure(figsize=(10, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=clustering)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output_file, bbox_inches="tight")
    plt.close()
    del X_scaled, Z


if __name__ == "__main__":
    main()

