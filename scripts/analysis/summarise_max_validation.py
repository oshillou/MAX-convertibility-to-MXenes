import argparse
import os
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from interprete import Capturing

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


def main():
    args = get_args()

    print("Loading labels")
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)
    y = (y>0).astype(int)

    positive_indices, = np.nonzero(y)

    print("Loading clustering")
    clusterings = np.zeros((len(args.clustering_files), len(y)), dtype=int)
    validation_index = np.zeros((len(args.clustering_files)), dtype=int)
    for i, filename in enumerate(args.clustering_files):
        clusterings[i] = pd.read_csv(filename, index_col=0)["Cluster"]
        validation_index[i] = int(filename.split("val")[-1].replace(".csv", ""))
    binary_clusterings = np.copy(clusterings)
    binary_clusterings[binary_clusterings>0] = 1

    # cluster 0 is always negative, others have one positive (synthesizable) material
    # largert: positive cluster with most samples
    # strongest: positive cluster with most non-masked synthesizable material
    number_of_positive_clusters = np.zeros(len(validation_index))
    number_of_positive_samples = np.zeros(len(validation_index))
    largert_cluster_sizes = np.zeros(len(validation_index))
    strongest_cluster_sizes = np.zeros(len(validation_index))
    strongest_cluster_counts = np.zeros(len(validation_index))
    big_successes = 0
    strong_sucesses = 0
    successes = 0

    with Capturing() as description:
        print("BEWARE: everything is written in 1-index, not 0-index [natural counting]")

        print("="*10,"\n","INDIVIDUAL METRICS","\n","="*10,"\n")
        print("When this (synthesizable) MAX phase was masked => it was clustered with: [MAX PHASE, MAX PHASE]")
        for i, index in enumerate(validation_index):
            max_phase = pu_labels.iloc[positive_indices[index]].name
            print(f"When {max_phase} was masked:")
            other_phases = pu_labels.iloc[positive_indices][clusterings[i, positive_indices]==clusterings[i, positive_indices[index]]].index
            other_phases = np.setxor1d(other_phases, max_phase)
            if len(other_phases)!=0:
                print(f"\t=> It ended in the same cluster as {other_phases}")
                is_success = True
            else:
                print(f"\t=> It did not end in a positive cluster")
                is_success = False

            cluster_sizes = np.bincount(clusterings[i])

            positive_cluster_sizes = cluster_sizes[1:]
            number_of_positive_clusters[i] = len(positive_cluster_sizes)
            number_of_positive_samples[i] = positive_cluster_sizes.sum()
            largert_cluster_sizes[i] = positive_cluster_sizes.max()
            largert_cluster_id = positive_cluster_sizes.argmax()+1
            positive_material_counts = np.bincount(clusterings[i, positive_indices])[1:]
            strongest_cluster_counts[i] = positive_material_counts.max()
            strongest_cluster_id = positive_material_counts.argmax()+1
            strongest_cluster_sizes[i] = positive_cluster_sizes[strongest_cluster_id-1]
            print(f"\t=> A total of {len(positive_cluster_sizes)} positive clusters were found")
            print(f"\t=> Their respective sizes were {positive_cluster_sizes} [taking into account all non-synthesizable materials]")
            print(f"\t=> The distribution of synthesizable MAX phases in these clusters is: {positive_material_counts}")
            if is_success:
                print(f"\t=> The MAX phase {max_phase} was in cluster {clusterings[i, positive_indices[index]]}")
                successes += 1
                if clusterings[i, positive_indices[index]] == largert_cluster_id:
                    big_successes += 1
                    print(f"\t=> This is a big success")
                if clusterings[i, positive_indices[index]] == strongest_cluster_id:
                    strong_sucesses += 1
                    print(f"\t=> This is a strong success")
        
        
        print("\n", "="*10,"\n","AVERAGE METRICS","\n","="*10,"\n")

        print(f"Out of {len(validation_index)} run(s), {successes} managed to bring back the masked material with another non-masked material")
        print(f"A total of {big_successes} run(s) had the masked sample in the largert positive clusters")
        print(f"A total of {strong_sucesses} run(s) had the masked sample in the positive cluster with most synthesizable materials")

        
        # Compute the average ARI for the entire batch
        total = 0
        binary_score, multiclass_score = 0, 0
        for i, j in itertools.combinations(range(len(clusterings)), r=2):
            total+=1
            multiclass_score += metrics.adjusted_rand_score(clusterings[i], clusterings[j])
            binary_score += metrics.adjusted_rand_score(binary_clusterings[i], binary_clusterings[j])

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

