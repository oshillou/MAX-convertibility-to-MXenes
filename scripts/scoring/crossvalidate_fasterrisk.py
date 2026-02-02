import argparse
import os
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection
from sklearn.utils.class_weight import compute_sample_weight
from functools import partial
import itertools
## Currently, fasterrisk uses the log loss, which is not adequate for imbalanced datasets
## Unfortunately, the log loss is hidden in the utils: https://github.com/interpretml/FasterRisk/blob/main/src/fasterrisk/utils.py
## and the current API does not allow to parameterise it

# So we have to shunt this function before importing the fasterrisk model
# To parametrise the weights, we make use of a global variable
sample_weights = np.zeros(10)
def weighted_logistic_loss(yXB):
    loss_terms = np.log(1.+np.exp(-yXB))
    loss_terms = loss_terms * sample_weights
    return loss_terms.sum() / sample_weights.sum()

import fasterrisk.utils
fasterrisk.utils.compute_logisticLoss_from_yXB = weighted_logistic_loss

from fasterrisk.binarization_util import convert_continuous_df_to_binary_df
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--results_file", type=str, default="interpretation.csv")
    parser.add_argument("--max_tests", type=int, default=30)

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)

    return args


def main():
    global sample_weights
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels>0]=1

    print("Preparing input data for Fasterrisk interpretation")
    X_binned = convert_continuous_df_to_binary_df(X).to_numpy()
    targets = 2*pseudo_labels-1

    splitter = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    number_models = 1

    results = []

    for number_tests in range(2, args.max_tests+1):
        print(f"Performing cross-validation for {number_tests} tests")
        for i, (train_indices, test_indices) in enumerate(splitter.split(X_binned, targets)):
            sample_weights = compute_sample_weight("balanced", pseudo_labels[train_indices])
            optimiser = RiskScoreOptimizer(X = X_binned[train_indices], y = targets[train_indices], k = number_tests, parent_size = number_models)

            print("Starting optimisation")
            optimiser.optimize()
            multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = optimiser.get_models()
            print("We generate {} risk score models from the sparse diverse pool".format(len(multipliers)))
            # Fake loop, we only explore the first model
            for multiplier, intercept, coefficients in zip(multipliers[:number_models], sparseDiversePool_beta0_integer[:number_models], sparseDiversePool_betas_integer[:number_models]):
                final_classifier = RiskScoreClassifier(multiplier, intercept, coefficients)
                y_pred = final_classifier.predict_prob(X_binned)

                local_result = {"tests":number_tests, "fold":i}

                scores = zip(["accuracy", "f1", "precision", "recall"],
                             [metrics.accuracy_score,
                              metrics.f1_score,
                              partial(metrics.precision_score, zero_division=0),
                              metrics.recall_score])

                for (name, indices), (score_name, score_fct) in itertools.product(zip(["train","test"], [train_indices, test_indices]), scores):
                    local_result[f"{name}_{score_name}"] = score_fct(pseudo_labels[indices], y_pred[indices]>0.5)

                for name, indices in zip(["train","test"], [train_indices, test_indices]):
                    local_result[f"{name}_auc"] = metrics.roc_auc_score(pseudo_labels[indices], y_pred[indices])
                    cmatrix = metrics.confusion_matrix(pseudo_labels[indices], y_pred[indices]>0.5, labels=[0,1])
                    local_result[f"{name}_tp"] = cmatrix[1,1]
                    local_result[f"{name}_tn"] = cmatrix[0,0]
                    local_result[f"{name}_fp"] = cmatrix[0,1]
                    local_result[f"{name}_fn"] = cmatrix[1,0]

            results += [local_result]
    results = pd.DataFrame(results)

    results.to_csv(args.results_file, index=False)

if __name__ == "__main__":
    main()
