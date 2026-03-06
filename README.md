# A clustering pipeline for hinting MAX phase convertibility to MXenes

## Global description

This repository contains an entire pipeline for identifying potential MXenes that could be converted from MAX precusors. The pipeline works in two steps:

1. A clustering algorithm is trained, and constraints are applied to maintain convertible MAX phases together
2. A final classifier is used to interprete the clusters and understand what distinguishes positive clusters (those with at least one convertible material) from the negative clusters. 

In this code, 4 different clustering algorithms are displayed: `KMeans`, `COPKMeans`, `PCKMeans` and `PUMKL`, which is a variant of `KernelSSC`.

The interpretable classifier at the end of the pipeline is `Fasterrisk`.

## Installation advice

To run this code, we used `Python 3.9`.

+ Please be careful to reinstall `matplotlib==3.2.2` after installing fasterrisk. The requirements in fasterrisk are too constraining, and this violation will not be harmful.
+ `pulp<2.8` is needed because a recent update made it [broken with snakemake](https://github.com/snakemake/snakemake/issues/2607)
+ `openpyxl` is needed for reading the excel sheet

=> note that installing `pulearn` switched scikit-learn from 1.2.0 to 1.5.2.

## Running the pipeline

The pipeline and all experiments can be executed at once using the command:

```commandline
snakemake all
```

This will execute both the main experiments and the validation experiments. This execution will create multiple folders:

+ `results` contains the raw clustering output of each clustering algorithm. Note that subfolders will be created for the different validation scenarios.
+ `analysis` contains the output of the `Fasterrisk` model applied on each clustering output, as well as summaries of the different validation scenarios.
+ `models` that stores all different model parameters after training for reusability purposes.


In case it does not work, we provide here the command lines step by step.

### Preparing the dataset

We already provide three files in the `data` folder containing the preprocessed dataset.

+ `data/categorical_data.csv` contains the preprocessed categorical features
+ `data/continuous_data.csv` contains the preprocessed continuous features
+ `data/pu_labels.csv` contains the pre-formatted labels for convertibility.

Any MAX phase that cannot be converted to an MXene is labeled 0. If a MAX phase can be converted, then its label is a positive integer that uniquely defines a specific MXene. For instance, *Ti2AlC* and *Ti2SC* have the same label (6) because they can be converted to *Ti2C*, whereas *V2AlC* has a different label (4) because V2C is a different MXene.

### Main run

To cluster the dataset according to a chosen algorithm, use:

```commandline
python scripts/clustering/main.py --seed 0 --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --n_clusters 150 --clustering_model pumkl --clustering_file results/pumkl_150.csv --model_file models/pumkl_150.pkl
```

Other available clustering algorithm options are `kmeans`, `copkmeans`, and `pckmeans`.

Once this was executed, the clustering output can be interpreted using Fasterrisk. To that end, a second script is provided. Simply execute:

```commandline
python scripts/analysis/interprete.py --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --clustering_file results/pumkl_150.csv --tsne_file analysis/pumkl_150_tsne.pdf --results_file analysis/pumkl_150_intepretation.txt --score_file analysis/pumkl_150_scores.xlsx --feature_importance analysis/pumkl_150_feature_importance.csv
```

This script returns 4 outputs:

+ A tsne projection of the clustering, with the samples coloured according to their belonging to positive/negative clusters
+ An interpretation file that describes the 10 different rule models that were produced by Fasterrisk. It is a plain text file to read.
+ An Excel file applying on each sheet its respective decision model and scoring the dataset.
+ A feature importance file (csv) describing the results of a feature importance analysis when fitting a random forest to the positive/negative clusters.

Of course, there are other ways of deriving **scores** from a clustering and map them to the notion of convertibility. One such straightforward way is via distances between samples and positive clusters.

### Validation runs

There are 4 main ways of doing validation in this pipeline, respectively called *MAX-inclusion*, *MAX-exclusion*, *MX-inclusion*, *MX-exclusion*.

In an *inclusion* setup, only the labels of a set of materials (or just one) are masked. The materials are still part of the training set both for the clustering algorithm and the Fasterrisk interpretation. In the *exclusion* scenario, that set of materials is held out, like an actual test set.

We can consider two different sets of compounds to act upon for validation. We can either mask/exclude a specific MAX phase precursor (*MAX* mode), or we can mask/exclude an entire set of MAX precursors that share a common MXene (*MX* mode).

#### MAX-inclusion setup

To exclude a MAX precusors, simply specify its number in the command line. Here, we mask the 3rd labeled MAX phase:

```commandline
python scripts/clustering/main.py --validate_max 2 --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --n_clusters 150 --clustering_model pumkl --clustering_file results/validate_max_include/pumkl_150_val2.csv --model_file models/validate_max_include/pumkl_150_val2.pkl
```

The same logic applies for interpretation. Simply add the `max_validation` flag:

```commandline
python scripts/analysis/interprete_validation.py --max_validation 2 other_params
```

For `other_params`, refer to the main interpretation script in the previous section.

After running the validation for all different 16 MAX precursors, the validation can be summarised using the following command:

```commandline
python scripts/analysis/summarise_max_validation.py --pu_labels data/pu_labels.csv --clustering_files results/validate_max_include/pumkl_150_val0.csv ... results/validate_max_include/pumkl_150_val15.csv --validation_file analysis/validate_max_include/pumkl_150_validation.txt --validation_heatmap analysis/validate_max_include/pumkl_150_connectivity.pdf
```

This will produce a text file summarising the validation metrics of this scenario and a heatmap showing how frequently each masked MAX phase was connected to the others.

#### MX-inclusion setup

For clustering, the options are very similar to MAX-inclusion and MAX-exclusion. The flag `validate_max` is replaced by `validate_mx`. Note that in the case of MX validation, the number to provide must match one of the positive integers in the label file `data/pu_labels.csv`

```commandline
python scripts/clustering/main.py --validate_mx 2 --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --n_clusters 150 --clustering_model pumkl --clustering_file results/validate_mx_include/pumkl_150_val2.csv --model_file models/validate_mx_include/pumkl_150_val2.pkl
```

For interpretation, simply add the `mx_validation` flag:

```commandline
python scripts/analysis/interprete_validation.py --mx_validation 2 other_params
```

For `other_params`, refer to the main interpretation script in the previous section.

After running the validation for all different 10 MXenes, the validation can be summarised using the following command:

```commandline
python scripts/analysis/summarise_mx_validation.py --pu_labels data/pu_labels.csv --clustering_files results/validate_mx_include/pumkl_150_val0.csv ... results/validate_max_include/pumkl_150_val9.csv --validation_file analysis/validate_mx_include/pumkl_150_validation.txt --validation_heatmap analysis/validate_mx_include/pumkl_150_connectivity.pdf
```

This will produce a text file summarising the validation metrics of this scenario and a heatmap showing how frequently each masked MAX phase was connected to the others.