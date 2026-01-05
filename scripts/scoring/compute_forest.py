import argparse
import os

import pandas as pd
import numpy as np
from sklearn import metrics, manifold, preprocessing, ensemble, inspection
from sklearn.utils.class_weight import compute_sample_weight
import sys
from io import StringIO
from ase.formula import Formula


class Capturing(list):
    # Code from: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--continuous_data", type=str, default="data/continuous_data.csv")
    parser.add_argument("--categorical_data", type=str, default="data/categorical_data.csv")
    parser.add_argument("--pu_labels", type=str, default="data/targets.csv")
    parser.add_argument("--clustering_file", type=str, default="clusters.csv")
    parser.add_argument("--results_file", type=str, default="interpretation.txt")
    parser.add_argument("--n_leaves", type=int, default=20)
    parser.add_argument("--score_file", type=str, default="score.xlsx")
    parser.add_argument("--max_validation", type=int, default=None)
    parser.add_argument("--mx_validation", type=int, default=None)
    parser.add_argument("--exclude_validation", default=False, action="store_true")

    args = parser.parse_args()

    assert os.path.exists(args.continuous_data)
    assert os.path.exists(args.categorical_data)
    assert os.path.exists(args.pu_labels)

    return args


def max2mx(max_name):
    max_counts = Formula(max_name).count()
    del max_counts[list(max_counts)[1]]  # A element is always in position 1
    mx_name_unsorted = "".join([f"{atom}{count}" for atom, count in max_counts.items()])
    return Formula(mx_name_unsorted).format("metal")


def main():
    global sample_weights
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels > 0] = 1

    train_indices = np.arange(len(X))
    validation_indices = []
    if args.max_validation is not None:
        validation_indices = np.array([np.nonzero(y)[0][args.max_validation]])
    elif args.mx_validation is not None:
        validation_indices = train_indices[y == args.mx_validation + 1]
    if len(validation_indices) != 0 and args.exclude_validation:
        train_indices = np.setxor1d(train_indices, validation_indices)

    # Now that we use the actual targets (with MXene groups) for computing the validation sets
    # We may convert it again to binary target for "(not) synthesizable"
    y = (y > 0).astype(int)

    print("Computing confusion matrix")
    full_cmatrix = metrics.confusion_matrix(y, clustering)
    binary_cmatrix = metrics.confusion_matrix(y, pseudo_labels)

    print("Computing sample weights")
    sample_weights = compute_sample_weight("balanced", pseudo_labels[train_indices])

    with Capturing() as output:
        print("Fitting a random forest to assess performances")
        random_forest = ensemble.RandomForestClassifier(max_leaf_nodes=args.n_leaves)
        random_forest.fit(X.iloc[train_indices], pseudo_labels[train_indices], sample_weight=sample_weights)
        y_prob = random_forest.predict_proba(X)
        y_pred = y_prob.argmax(1)
        if not args.exclude_validation:
            print("Accuracy of random forest with pseudo-labels:", metrics.accuracy_score(pseudo_labels, y_pred))
            print("F1 score with pseudo-labels", metrics.f1_score(pseudo_labels, y_pred))
            print("AUC with pseudo-labels:", metrics.roc_auc_score(pseudo_labels, y_prob[:,1]))
            print("Confusion matrix between random forest and pseudo-labels")
            print(metrics.confusion_matrix(pseudo_labels, y_pred))
            print("Same confusion matrix restricted to positive samples")
            print(metrics.confusion_matrix(pseudo_labels[y==1], y_pred[y==1]))
            print("Synthesizable MAX phases that were predicted as negative by this model")
            print(X[(y==1)&(y_pred==0.5)])
        else:
            for name, idx in zip(["Train", "Val"], [train_indices, validation_indices]):
                print(f"{name} scores", "-"*10)
                print("Accuracy with pseudo-labels:", metrics.accuracy_score(pseudo_labels[idx], y_pred[idx]>0.5))
                if name=="Train":
                    print("F1 score with pseudo-labels", metrics.f1_score(pseudo_labels[idx], y_pred[idx]>0.5))
                    print("AUC with pseudo-labels:", metrics.roc_auc_score(pseudo_labels[idx], y_pred[idx]))
                    print("Confusion matrix between predictions and pseudo-labels")
                    print(metrics.confusion_matrix(pseudo_labels[idx], y_pred[idx]>0.5))
                    print("Same confusion matrix restricted to positive samples")
                    print(metrics.confusion_matrix(pseudo_labels[idx][y[idx]==1], y_pred[idx][y[idx]==1]>0.5))
                else:
                    print("Accuracy with ground PU truth", metrics.accuracy_score(y[idx], y_pred[idx]>0.5))
                print("Synthesizable MAX phases that were predicted as negative by this model")
                print(X.iloc[idx][(y[idx]==1)&(y_pred[idx]<=0.5)])

        print("Performing feature importance")
        # Finally, perform feature importance
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        feature_importance = inspection.permutation_importance(random_forest,
                                                               X.iloc[train_indices],
                                                               pseudo_labels[train_indices],
                                                               n_repeats=20,
                                                               random_state=0,
                                                               sample_weight=sample_weights)


        print("Top 20 feature importances are:")
        scores = feature_importance.importances_mean
        print(X.columns[np.argsort(scores)][-20:])
        print(scores[np.argsort(scores)][-20:])


    print("Exporting Excel file of dataset scores")
    with pd.ExcelWriter(args.score_file) as excel_writer:
        columns = {
            "Total Score": y_prob[:, 1],
            "Original cluster": clustering,
            "MXene": [max2mx(name) for name in X.index]
        }
        scored_df = pd.DataFrame(index=X.index, data=columns)
        scored_df.to_excel(excel_writer, sheet_name="Model outputs")

        columns = {
            "importance":feature_importance.importances_mean,
            "std":feature_importance.importances_std
        }
        scored_df = pd.DataFrame(index=X.columns, data=columns)
        scored_df.to_excel(excel_writer, sheet_name="Feature importance")

    print("Exporting description")
    description = [f"Multiclass confusion matrix:\n{full_cmatrix}", f"Pseudo-label confusion matrix:\n{binary_cmatrix}",
                   "Random forest performance:\n" + "\n".join(output)]
    for k in sorted(np.unique(clustering)):
        if k == 0:
            continue
        description.append(f"Compounds in cluster {k}")
        for name in X.index[(clustering == k) & (y == 1)]:
            description.append(f"\t=> {name}")

    with open(args.results_file, "w") as file:
        file.write("\n".join(description))


if __name__ == "__main__":
    main()

