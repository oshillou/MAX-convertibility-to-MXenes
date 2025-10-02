import argparse
import os
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--results_file", type=str, default="interpretation.csv")

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
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)
    y = (y>0).astype(int)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels>0]=1

    print("Preparing input data for Fasterrisk interpretation")
    X_binned = convert_continuous_df_to_binary_df(X).to_numpy()
    targets = 2*pseudo_labels-1

    splitter = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    number_models = 1

    results = []

    for number_features in range(2, 10+1):
        print(f"Performing cross-validation for {number_features} features")
        for i, (train_indices, test_indices) in enumerate(splitter.split(X_binned, targets)):
            optimiser = RiskScoreOptimizer(X = X_binned[train_indices], y = targets[train_indices], k = number_features, parent_size = number_models)

            print("Starting optimisation")
            optimiser.optimize()
            multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = optimiser.get_models()
            print("We generate {} risk score models from the sparse diverse pool".format(len(multipliers)))
            for multiplier, intercept, coefficients in zip(multipliers[:number_models], sparseDiversePool_beta0_integer[:number_models], sparseDiversePool_betas_integer[:number_models]):
                final_classifier = RiskScoreClassifier(multiplier, intercept, coefficients)
                y_pred_train = final_classifier.predict_prob(X_binned[train_indices])
                y_pred_test = final_classifier.predict_prob(X_binned[test_indices])

                local_result = {"features":number_features, "fold":i}

                local_result["train_accuracy"] = metrics.accuracy_score(pseudo_labels[train_indices], y_pred_train>0.5)
                local_result["test_accuracy"] = metrics.accuracy_score(pseudo_labels[test_indices], y_pred_test>0.5)
                local_result["train_f1"] = metrics.f1_score(pseudo_labels[train_indices], y_pred_train>0.5)
                local_result["test_f1"] = metrics.f1_score(pseudo_labels[test_indices], y_pred_test>0.5)
                local_result["train_auc"] = metrics.roc_auc_score(pseudo_labels[train_indices], y_pred_train)
                local_result["test_auc"] = metrics.roc_auc_score(pseudo_labels[test_indices], y_pred_test)
            results += [local_result]
    results = pd.DataFrame(results)

    results.to_csv(args.results_file, index=False)

if __name__ == "__main__":
    main()

