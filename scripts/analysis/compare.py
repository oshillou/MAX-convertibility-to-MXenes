import argparse
import os
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--clustering_files", type=str, nargs="+", required=True)

    parser.add_argument("--multiclass_ari_plot", type=str, default="multiclass_ari.pdf")
    parser.add_argument("--binaryclass_ari_plot", type=str, default="binaryclass_ari.pdf")
    parser.add_argument("--connectivity_plot", type=str, default="connectivity_plot.pdf")

    args = parser.parse_args()

    assert os.path.exists(args.pu_labels)

    for filename in args.clustering_files:
        assert os.path.exists(filename)

    return args


def main():
    args = get_args()

    print("Loading labels")
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)

    positive_indices, = np.nonzero(y)

    print("Loading clustering")
    clusterings = np.zeros((len(args.clustering_files), len(y)), dtype=int)
    for i, filename in enumerate(args.clustering_files):
        clusterings[i] = pd.read_csv(filename, index_col=0)["Cluster"]
    binary_clusterings = np.copy(clusterings)
    binary_clusterings[binary_clusterings>0] = 1

    print("Computing ARI scores")
    # Compute the average ARI for the entire batch
    multiclass_ari_scores = np.eye((len(clusterings)))
    binary_ari_scores = np.eye((len(clusterings)))
    for i, j in itertools.combinations(range(len(clusterings)), r=2):
        multiclass_ari_scores[i,j] = metrics.adjusted_rand_score(clusterings[i], clusterings[j])
        multiclass_ari_scores[j,i] = multiclass_ari_scores[i,j]
        binary_ari_scores[i,j] = metrics.adjusted_rand_score(binary_clusterings[i], binary_clusterings[j])
        binary_ari_scores[j,i] = binary_ari_scores[i,j]

    print("Computing matrix of average co-clustering for positive compounds (substructures)")
    positive_clustering = clusterings[:, positive_indices]
    compound_names = pu_labels.iloc[positive_indices].index
    model_names = [x.split(os.sep)[-1].split("_")[0] for x in args.clustering_files]

    connectivity_matrix = np.eye(len(compound_names))
    for i, j in itertools.combinations(range(len(compound_names)), r=2):
        connectivity_matrix[i,j] = (positive_clustering[:,i]==positive_clustering[:,j]).mean()
        connectivity_matrix[j,i] = connectivity_matrix[i,j]
    
    print("Exporting multi-class ARI matrix")
    sns.heatmap(multiclass_ari_scores, vmin=0, vmax=1, cmap="seismic", annot=True, fmt=".2f", cbar=False)
    plt.xlabel("Clustering model")
    plt.xticks(ticks=np.arange(len(model_names))+0.5, labels=model_names, rotation=60)
    plt.ylabel("Clustering model")
    plt.yticks(ticks=np.arange(len(model_names))+0.5, labels=model_names, rotation=0)
    plt.tight_layout()
    plt.savefig(args.multiclass_ari_plot, bbox_inches="tight")
    plt.close()

    print("Exporting binary class ARI matrix")
    sns.heatmap(binary_ari_scores, vmin=0, vmax=1, cmap="seismic", annot=True, fmt=".2f", cbar=False)
    plt.xlabel("Clustering model")
    plt.xticks(ticks=np.arange(len(model_names))+0.5, labels=model_names, rotation=60)
    plt.ylabel("Clustering model")
    plt.yticks(ticks=np.arange(len(model_names))+0.5, labels=model_names, rotation=0)
    plt.tight_layout()
    plt.savefig(args.binaryclass_ari_plot, bbox_inches="tight")
    plt.close()

    print("Exporting the connectivity matrix for positive samples in multiclass context")
    sns.heatmap(connectivity_matrix, vmin=0, vmax=1, cmap="seismic", annot=True, fmt=".2f", cbar=False)
    plt.xlabel("Compound")
    plt.xticks(ticks=np.arange(len(compound_names))+0.5, labels=compound_names, rotation=60)
    plt.ylabel("Compound")
    plt.yticks(ticks=np.arange(len(compound_names))+0.5, labels=compound_names, rotation=0)
    plt.tight_layout()
    plt.savefig(args.connectivity_plot, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

