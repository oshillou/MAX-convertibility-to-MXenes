import argparse
import os

import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, ensemble, svm, preprocessing
from sklearn.utils.class_weight import compute_sample_weight
from functools import partial
import itertools
from compute_svc import pumkl_kernel

import sys
sys.path.append(os.path.join("scripts/clustering"))
from pu_clustering import *

import pickle
from functools import partial


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--model_file", type=str, default="model.pkl")
    parser.add_argument("--results_file", type=str, default="interpretation.csv")
    parser.add_argument("--max_slack", type=int, default=3, help="Expressed in exponent form")

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)
    assert os.path.exists(args.clustering_file)
    assert os.path.exists(args.model_file)

    return args


def main():
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    # In the case of SVC, we need to maintain the preprocessing in order to use the original kernels
    # that were used by pumkl
    # In contrast, fasterrisk and random forests are insensitive to scaling (only ordering matters)
    X_cont = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X_cont), columns=X_cont.columns, index=X_cont.index)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels>0]=1

    print("Loading kernel matrix")
    with open(args.model_file, "rb") as file:
        clustering_model = pickle.load(file)
    if "PUMKL" in str(type(clustering_model)):
        # We are dealing with a custom kernel
        # So, we define a callable for it
        kernel_fct = partial(pumkl_kernel, clustering_model=clustering_model)

    else:
        kernel_fct = metrics.pairwise.linear_kernel # Simple linear kernel (Euclidean distance)


    splitter = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    results = []

    for C_value in np.logspace(-3, args.max_slack):
        print(f"Performing cross-validation for slack C={C_value}")
        for i, (train_indices, test_indices) in enumerate(splitter.split(X, pseudo_labels)):
            sample_weights = compute_sample_weight("balanced", pseudo_labels[train_indices])

            svc_model = svm.SVC(C=C_value, kernel=kernel_fct, probability=True)

            svc_model.fit(X.iloc[train_indices], pseudo_labels[train_indices], sample_weight=sample_weights)
            y_pred = svc_model.predict_proba(X)

            local_result = {"C":C_value, "fold":i}

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
