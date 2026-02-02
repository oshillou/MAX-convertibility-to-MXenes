import argparse
import os

import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, ensemble
from sklearn.utils.class_weight import compute_sample_weight
from functools import partial
import itertools


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--results_file", type=str, default="interpretation.csv")
    parser.add_argument("--max_leaves", type=int, default=40)

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)

    return args


def main():
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels>0]=1


    splitter = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    results = []

    for n_leaves in range(2, args.max_leaves+1):
        print(f"Performing cross-validation for {n_leaves} leaves")
        for i, (train_indices, test_indices) in enumerate(splitter.split(X, pseudo_labels)):
            sample_weights = compute_sample_weight("balanced", pseudo_labels[train_indices])

            model = ensemble.RandomForestClassifier(max_leaf_nodes=n_leaves)

            model.fit(X.iloc[train_indices], pseudo_labels[train_indices], sample_weight=sample_weights)
            y_pred = model.predict_proba(X)

            local_result = {"leaves":n_leaves, "fold":i}

            scores = zip(["accuracy", "f1", "precision", "recall"],
                         [metrics.accuracy_score,
                          metrics.f1_score,
                          partial(metrics.precision_score, zero_division=0),
                          metrics.recall_score])

            for (name, indices), (score_name, score_fct) in itertools.product(zip(["train","test"], [train_indices, test_indices]), scores):
                local_result[f"{name}_{score_name}"] = score_fct(pseudo_labels[indices], y_pred[indices].argmax(1))

            for name, indices in zip(["train","test"], [train_indices, test_indices]):
                local_result[f"{name}_auc"] = metrics.roc_auc_score(pseudo_labels[indices], y_pred[:,1][indices])
                cmatrix = metrics.confusion_matrix(pseudo_labels[indices], y_pred[indices].argmax(1), labels=[0,1])
                local_result[f"{name}_tp"] = cmatrix[1,1]
                local_result[f"{name}_tn"] = cmatrix[0,0]
                local_result[f"{name}_fp"] = cmatrix[0,1]
                local_result[f"{name}_fn"] = cmatrix[1,0]

            results += [local_result]
    results = pd.DataFrame(results)

    results.to_csv(args.results_file, index=False)

if __name__ == "__main__":
    main()
