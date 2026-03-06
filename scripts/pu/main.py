import pandas as pd
import pulearn
from sklearn import preprocessing, model_selection, metrics, tree
import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--T", type=int, default=100, help="Number of weak learners in PU bagging")
    parser.add_argument("--k", type=int, default=5, help="Number of positive labels kept per bootstrap")
    parser.add_argument("--R", type=int, default=200, help="Number of bootstrap repetitions")
    parser.add_argument("--scoring_file", type=str, default="scores.csv")

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)
    assert os.path.exists(args.pu_labels)

    return args


def main():
    args = get_args()

    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cont = preprocessing.MinMaxScaler().fit_transform(X_cont)
    X_cat = pd.read_csv(args.categorical_data, index_col=0).to_numpy()
    X = np.concatenate([X_cont, X_cat], axis=1)
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = (pu_labels["class"]>0).to_numpy().astype(int)

    if args.k>sum(y):
        print("Changing k to the number of targets", f"{args.k}>{sum(y)}")
        args.k = sum(y)

    R = 200  # Number of repeats (to get sub-estimates for lower R, subsampling would suffice)

    feature_indices = [f"feature_{i}" for i in range(X.shape[1])]
    sample_indices = [f"max_{i}" for i in range(len(X))]

    splitter = model_selection.RepeatedStratifiedKFold(n_splits=args.k, n_repeats=R, random_state=0)

    all_outputs = []
    base_estimator = tree.DecisionTreeClassifier(class_weight="balanced")
    pu_estimator = pulearn.BaggingPuClassifier(estimator=base_estimator, n_estimators=args.T, max_samples=args.k,
                                               max_features=X.shape[1], n_jobs=8)

    for r, (train_indices, test_indices) in enumerate(splitter.split(X, y)):

        # Fit
        pu_estimator.fit(X[train_indices], y[train_indices])

        # We now must get the predictions on the test set and the training set
        y_pred = np.zeros(len(y))
        y_pred[train_indices] = pu_estimator.oob_decision_function_[:, 1]
        y_pred[test_indices] = pu_estimator.predict_proba(X[test_indices])[:, 1]

        # Now, we must estimate the recall on the test set, and the PU score
        tpr = metrics.recall_score(y[test_indices], y_pred[test_indices].round())
        pu_score = tpr ** 2 / (1e-12 + y_pred[test_indices].round().mean())

        feature_importances = sum([x.feature_importances_ for x in pu_estimator.estimators_])

        feature_importances = pd.Series(data=feature_importances, index=feature_indices)
        probs = pd.Series(data=y_pred, index=sample_indices)
        data_row = pd.concat([feature_importances, probs])
        data_row["Split"] = r%args.k
        data_row["Repeat"] = r//args.k
        data_row["TPR"] = tpr
        data_row["PU_score"] = pu_score

        all_outputs += [data_row]

    print("Finished running, exporting to", args.scoring_file)
    pd.DataFrame(all_outputs).to_csv(args.scoring_file)

if __name__ == "__main__":
    main()