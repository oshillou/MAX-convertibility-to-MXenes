import argparse
import os
import pandas as pd
import numpy as np
from sklearn import metrics, manifold, preprocessing, ensemble, inspection
from sklearn.utils.class_weight import compute_sample_weight
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
from fasterrisk.utils import get_support_indices
from matplotlib import pyplot as plt
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
    parser.add_argument("--feature_importance", type=str, default="feature_importance.csv")
    parser.add_argument("--n_tests", type=int, default=20)
    parser.add_argument("--score_file", type=str, default="score.xlsx")
    parser.add_argument("--tsne_file", type=str, default="tsne.pdf")
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
    del max_counts[list(max_counts)[1]] # A element is always in position 1
    mx_name_unsorted = "".join([f"{atom}{count}" for atom, count in max_counts.items()])
    return Formula(mx_name_unsorted).format("metal")

def main():
    args = get_args()

    print("Loading data")
    X_cont = pd.read_csv(args.continuous_data, index_col=0)
    X_cat = pd.read_csv(args.categorical_data, index_col=0)
    X = pd.concat([X_cont, X_cat], axis=1)
    pu_labels = pd.read_csv(args.pu_labels, index_col=0)
    y = pu_labels.to_numpy().reshape(-1)

    clustering = pd.read_csv(args.clustering_file, index_col=0).to_numpy().reshape(-1)
    pseudo_labels = np.copy(clustering)
    pseudo_labels[pseudo_labels>0]=1

    train_indices = np.arange(len(X))
    validation_indices = np.arange(0)
    if args.max_validation is not None:
        validation_indices = np.array([np.nonzero(y)[0][args.max_validation]])
    elif args.mx_validation is not None:
        validation_indices = train_indices[y==args.mx_validation+1]
    if len(validation_indices) != 0 and args.exclude_validation:
        train_indices = np.setxor1d(train_indices, validation_indices)

    # Now that we use the actual targets (with MXene groups) for computing the validation sets
    # We may convert it again to binary target for "(not) synthesizable"
    y = (y>0).astype(int)

    print("Computing and exporting TSNE")
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    Z = manifold.TSNE().fit_transform(X_scaled)
    plt.figure(figsize=(10,5))
    plt.scatter(Z[:,0], Z[:,1], c=clustering)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.tsne_file, bbox_inches="tight")
    plt.close()
    del X_scaled, Z

    
    print("Computing confusion matrix")
    full_cmatrix = metrics.confusion_matrix(y, clustering)[:2]
    binary_cmatrix = metrics.confusion_matrix(y, pseudo_labels)


    print("Preparing input data for Fasterrisk interpretation")
    X_binned = convert_continuous_df_to_binary_df(X)
    targets = 2*pseudo_labels-1

    number_models = 1

    optimiser = RiskScoreOptimizer(X = X_binned.to_numpy()[train_indices], y = targets[train_indices], k = args.n_tests, parent_size = number_models)

    print("Starting optimisation")
    optimiser.optimize()
    multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = optimiser.get_models()
    print("We generate {} risk score models from the sparse diverse pool".format(len(multipliers)))
    all_scored_dataframes = []
    all_pseudo_predictions = []
    with Capturing() as output:
        for multiplier, intercept, coefficients in zip(multipliers[:number_models], sparseDiversePool_beta0_integer[:number_models], sparseDiversePool_betas_integer[:number_models]):
            final_classifier = RiskScoreClassifier(multiplier, intercept, coefficients)
            y_pred = final_classifier.predict_prob(X_binned.to_numpy())
            final_classifier.reset_featureNames(X_binned.columns)

            final_classifier.print_model_card()
        
            if not args.exclude_validation:
                print("Accuracy with pseudo-labels:", metrics.accuracy_score(pseudo_labels, y_pred>0.5))
                print("F1 score with pseudo-labels", metrics.f1_score(pseudo_labels, y_pred>0.5))
                print("AUC with pseudo-labels:", metrics.roc_auc_score(pseudo_labels, y_pred))
                print("Confusion matrix between predictions and pseudo-labels")
                print(metrics.confusion_matrix(pseudo_labels, y_pred>0.5))
                print("Same confusion matrix restricted to positive samples")
                print(metrics.confusion_matrix(pseudo_labels[y==1], y_pred[y==1]>0.5))
                print("Synthesizable MAX phases that were predicted as negative by this model")
                print(X[(y==1)&(y_pred<=0.5)])

                print("Top 10 candidates for synthesizability")
                candidates = np.arange(len(X))[y==0]
                order = np.argsort(y_pred[candidates])
                sorted_candidates = candidates[order]

                print("Scores: ", y_pred[sorted_candidates[-10:]])
                print(X.iloc[sorted_candidates[-10:]])
                print("\n\n")
            else:
                for name, idx in zip(["Train", "Val"], [train_indices, validation_indices]):
                    print(f"{name} scores", "-"*10)
                    print("Accuracy with pseudo-labels:", metrics.accuracy_score(pseudo_labels[idx], y_pred[idx]>0.5))
                    if name == "Train":
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


            # Finally, store the scores for this classifier
            support_indices = get_support_indices(final_classifier.coefficients)
            scored_df = X_binned[X_binned.columns[support_indices]]
            scored_df = scored_df * final_classifier.coefficients[support_indices].astype(int).reshape((1,-1))
            scored_df["Total score"] = scored_df.sum(axis=1)
            scored_df["Probability of pseudo-positivity"] = y_pred
            scored_df["Original cluster"] = clustering
            scored_df["MXene"] = [max2mx(name) for name in X.index]
            scored_df.index=X.index
            all_scored_dataframes += [scored_df]

            all_pseudo_predictions += [y_pred]

    print("Exporting Excel file of dataset scores")
    with pd.ExcelWriter(args.score_file) as excel_writer:
        for i, df in enumerate(all_scored_dataframes):
            df.to_excel(excel_writer, sheet_name=f"Model {i:>2}")

    with Capturing(output) as output:
        print("Fitting a random forest to assess performances")
        random_forest = ensemble.RandomForestClassifier()
        random_forest.fit(X.iloc[train_indices], pseudo_labels[train_indices], sample_weight=compute_sample_weight("balanced",pseudo_labels[train_indices]))
        y_prob = random_forest.predict_proba(X)
        y_pred = y_prob.argmax(1)
        if not args.exclude_validation:
            print("Accuracy of random forest with pseudo-labels:", metrics.accuracy_score(pseudo_labels, y_pred))
            print("F1 score with pseudo-labels", metrics.f1_score(pseudo_labels, y_pred>0.5))
            print("AUC with pseudo-labels:", metrics.roc_auc_score(pseudo_labels, y_prob[:,1]))
            print("Confusion matrix between random forest and pseudo-labels")
            print(metrics.confusion_matrix(pseudo_labels, y_pred))
            print("Same confusion matrix restricted to positive samples")
            print(metrics.confusion_matrix(pseudo_labels[y==1], y_pred[y==1]))
            print("Synthesizable MAX phases that were predicted as negative by this model")
            print(X[(y==1)&(y_pred<=0.5)])
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

        # Finally, perform feature importante 
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        feature_importance = inspection.permutation_importance(random_forest,
                                                               X.iloc[train_indices],
                                                               pseudo_labels[train_indices],
                                                               n_repeats=args.n_tests,
                                                               random_state=0,
                                                               sample_weight=compute_sample_weight("balanced", pseudo_labels[train_indices]))
        print("Feature importances are:")
        feature_importance = pd.Series(feature_importance.importances_mean, index=X.columns)
        print(feature_importance)
        feature_importance.to_csv(args.feature_importance)



    print("Exporting description")
    description = []
    description.append(f"Multiclass confusion matrix:\n{full_cmatrix}")
    description.append(f"Pseudo-label confusion matrix:\n{binary_cmatrix}")
    description.append("Interpretation cards:\n"+"\n".join(output))
    for k in sorted(np.unique(clustering)):
        if k==0:
            continue
        description.append(f"Compounds in cluster {k}")
        for name in X.index[(clustering==k)&(y==1)]:
            description.append(f"\t=> {name}")

    with open(args.results_file, "w") as file:
        file.write("\n".join(description))
    

if __name__ == "__main__":
    main()

