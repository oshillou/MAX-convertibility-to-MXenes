import argparse
import os
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from interprete import Capturing
from ase.formula import Formula

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--clustering_files", type=str, nargs="+", required=True)

    parser.add_argument("--validation_file", type=str, default="validation_analysis.txt")
    parser.add_argument("--validation_heatmap", type=str, default="heatmap.png")

    args = parser.parse_args()

    assert os.path.exists(args.pu_labels)

    for filename in args.clustering_files:
        assert os.path.exists(filename)

    return args

def max2mx(max_name):
    max_counts = Formula(max_name).count()
    del max_counts[list(max_counts)[1]] # A element is always in position 1
    mx_name_unsorted = "".join([f"{atom}{count}" for atom, count in max_counts.items()])
    return Formula(mx_name_unsorted).format("metal")

def main():
    args = get_args()

    print("Loading labels")
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)

    positive_indices, = np.nonzero(y)

    print("Loading clustering")
    clusterings = np.zeros((len(args.clustering_files), len(y)), dtype=int)
    validation_index = np.zeros((len(args.clustering_files)), dtype=int)
    for i, filename in enumerate(args.clustering_files):
        clusterings[i] = pd.read_csv(filename, index_col=0)["Cluster"]
        validation_index[i] = int(filename.split("val")[-1].replace(".csv", ""))
    binary_clusterings = np.copy(clusterings)
    binary_clusterings[binary_clusterings>0] = 1

    cluster_sizes = [np.bincount(x) for x in clusterings]
    masks = np.zeros((len(args.clustering_files), len(positive_indices)))
    for i, group in enumerate(y[positive_indices]):
        masks[group-1, i] = 1
    largert_cluster_number = np.array([x[1:].argmax()+1 for x in cluster_sizes])
    largert_cluster_sizes = np.array([cluster_sizes[i][x] for i, x in enumerate(largert_cluster_number)])

    positive_samples_clusterings = clusterings[:, positive_indices]
    synthesizable_cluster_sizes = [np.bincount(x) for x in positive_samples_clusterings]
    strongest_cluster_number = np.array([x[1:].argmax()+1 for x in synthesizable_cluster_sizes])
    strongest_cluster_sizes = np.array([cluster_sizes[i][x] for i, x in enumerate(strongest_cluster_number)])

    strongest_cluster_counts = np.array([synthesizable_cluster_sizes[i][x] for i, x in enumerate(strongest_cluster_number)])

    with Capturing() as description:
        print("BEWARE: everything is written in 1-index, not 0-index [natural counting]")

        print("="*10,"\n","INDIVIDUAL METRICS","\n","="*10,"\n")
        for i, index in enumerate(validation_index):
            masked_materials = pu_labels[pu_labels["class"]==index+1].index
            mxenes = [max2mx(name) for name in masked_materials]
            print(f"When the MAX phases {masked_materials} are masked, thus masking the MXenes {mxenes}")

            print(f"\tNumber of positive clusters: {len(cluster_sizes[i])-1}")
            print(f"\tNumber of samples clustered as positive: {cluster_sizes[i][1:].sum()}")

            for k in range(len(cluster_sizes[i])):
                print(f"\tCluster {k}", end=" ")
                if k==0:
                    print("(negative)", end=" ")
                if largert_cluster_number[i] == k:
                    print("(largert)", end=" ")
                if strongest_cluster_number[i] == k:
                    print("(strongest)", end=" ")
                print(f"/ {cluster_sizes[i][k]} samples")
                
                for j, name in enumerate(["Unmasked", "Masked"]):
                    materials = positive_indices[(positive_samples_clusterings[i]==k)&(masks[i]==j)]
                    if len(materials) != 0:
                        material_names = ", ".join(pu_labels.iloc[materials].index)
                        print(f"\t\t{name} :", material_names)
        
        
        
        print("\n", "="*10,"\n","AVERAGE METRICS","\n","="*10,"\n")

        positive_accuracy = [binary_clusterings[i, positive_indices[masks[i]==1]].mean() for i in range(len(validation_index))]
        largert_accuracy = [(clusterings[i, positive_indices[masks[i]==1]]==largert_cluster_number[i]).mean() for i in range(len(validation_index))]
        strongest_accuracy = [(clusterings[i, positive_indices[masks[i]==1]]==strongest_cluster_number[i]).mean() for i in range(len(validation_index))]

        print(f"Out of {len(validation_index)} run(s), the accuracy for setting back masked samples among positive clusters was: {np.mean(positive_accuracy):.4g} +/- {np.std(positive_accuracy):.4g}")
        print(f"For the largert cluster, this accuracy was: {np.mean(largert_accuracy):.4g} +/- {np.std(largert_accuracy):.4g}")
        print(f"For the strongest cluster, this accuracy was: {np.mean(strongest_accuracy):.4g} +/- {np.std(strongest_accuracy):.4g}")

        
        # Compute the average ARI for the entire batch
        total = 0
        binary_score, multiclass_score = 0, 0
        for i, j in itertools.combinations(range(len(clusterings)), r=2):
            total+=1
            multiclass_score += metrics.adjusted_rand_score(clusterings[i], clusterings[j])
            binary_score += metrics.adjusted_rand_score(binary_clusterings[i], binary_clusterings[j])

        number_of_positive_clusters = np.array([len(cluster_sizes[i][1:]) for i in range(len(cluster_sizes))])
        number_of_positive_samples = np.array([cluster_sizes[i][1:].sum() for i in range(len(cluster_sizes))])
        print(f"The average number of positive clusters is: {number_of_positive_clusters.mean():.4g} (STD: {number_of_positive_clusters.std():.4g})")
        print(f"The average number of samples that are part of a positive cluster is: {number_of_positive_samples.mean():.4g} (STD: {number_of_positive_samples.std():.4g})")
        print(f"The average number of samples in the largert positive cluster is: {largert_cluster_sizes.mean():.4g} (STD: {largert_cluster_sizes.std():.4g})")
        print(f"The average number of samples in the cluster that has the most syntheiszable materials is: {strongest_cluster_sizes.mean():.4g} (STD: {strongest_cluster_sizes.std():.4g})")
        print(f"The average number of synthesizable materials in the cluste with the most synthesizable materials is: {strongest_cluster_counts.mean():.4g} (STD: {strongest_cluster_counts.std():.4g})")

        print(f"The average ARI (agreement measure in [0,1]) between all runs while taking into account substructures is: {multiclass_score:.4g}")
        print(f"The average ARI between all runs when merging all positive clusters together is: {binary_score:.4g}")

    with open(args.validation_file, "w") as file:
        file.write("\n".join(description))

    connectivity_matrix = np.eye(len(positive_indices), dtype=int) * len(positive_indices)
    for i, j in itertools.combinations(range(len(positive_indices)), r=2):
        connectivity_matrix[i,j] = np.sum(clusterings[:,positive_indices[i]]==clusterings[:,positive_indices[j]])
        connectivity_matrix[j,i] = connectivity_matrix[i,j]

    materials = [pu_labels.iloc[index].name for index in positive_indices]
    plt.figure(figsize=(10,10))
    sns.heatmap(connectivity_matrix, annot=True, vmin=0, vmax=len(materials))
    plt.xlabel("Synthesizable material")
    plt.xticks(np.arange(len(positive_indices))+0.5, materials, rotation=45)
    plt.ylabel("Synthesizable material")
    plt.yticks(np.arange(len(positive_indices))+0.5, materials, rotation=0)
    plt.tight_layout()
    plt.savefig(args.validation_heatmap, bbox_inches="tight")
    plt.close()
    

if __name__ == "__main__":
    main()

