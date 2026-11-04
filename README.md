# A clustering pipeline for hinting MAX phase convertibility to MXenes

## Global description

This repository contains an entire pipeline for identifying potential MXenes that could be converted from MAX precusors. The pipeline works in two steps:

1. A clustering algorithm is trained, and constraints are applied to maintain convertible MAX phases together
2. A final classifier is used to interprete the clusters and understand what distinguishes positive clusters (those with at least one convertible material) from the negative clusters. 

In this code, 4 different clustering algorithms are displayed: `KMeans`, `COPKMeans`, `PCKMeans` and `PUMKL`, which is a variant of `KernelSSC`.

The interpretable classifier at the end of the pipeline is `SVC`.

## Direct results

We provide the Excel sheet with scores and original cluster from all 2,262 MAX precursors considered in our paper. The sheet can be found at `results/scoring/svc_5/pumkl/`. Or click [here](results/scoring/svc_5/pumkl/svc_pumkl_150_scores.xlsx).

## Installation advice

To run this code, we used `Python 3.9`.

+ `pulp<2.8` is needed because a recent update made it [broken with snakemake](https://github.com/snakemake/snakemake/issues/2607)
+ `openpyxl` is needed for reading the excel sheet

## Running the pipeline

The pipeline and all experiments can be executed at once using the command:

```commandline
snakemake all
```

This will execute both the main experiments and the validation experiments. This execution will create multiple folders:

+ `results/clustering` contains the raw clustering output of each clustering algorithm. Note that subfolders will be created for the different validation scenarios.
+ `results/scoring` contains the output of the scoring model applied on each clustering output.
+ `results/visualisation` contains t-SNEs of the different clusterings.
+ `validation` will contain the validation outcomes for the different setups: timeline validation, cros-validation of the scoring method, MAX or MX masked out. Each subfolder follows approximately the same structure as `results`.


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

Once this is executed, the clustering output can be scored `SVC`. To that end, a second script is provided. Simply execute:

```commandline
python scripts/scoring/compute_svc.py --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --clustering_file results/pumkl_150.csv --results_file analysis/pumkl_150_intepretation.txt --score_file analysis/pumkl_150_scores.xlsx
```

This script returns 1 output: the convertibility score of each MAX phase to an MXene.

### Validation runs

There are 2 main ways of doing validation in this pipeline, respectively called *MAX-inclusion*, *MX-inclusion*.

Only the labels of a set of materials (or just one) are masked. The materials are still part of the training set both for the clustering algorithm and the scoring.

We can consider two different sets of compounds to act upon for validation. We can either mask a specific MAX phase precursor (*MAX* mode), or we can mask an entire set of MAX precursors that share a common MXene (*MX* mode).

#### MAX-inclusion setup

To mask a MAX precusor, simply specify its number in the command line. Here, we mask the 3rd labeled MAX phase:

```commandline
python scripts/clustering/main.py --validate_max 2 --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --n_clusters 150 --clustering_model pumkl --clustering_file results/validate_max_include/pumkl_150_val2.csv --model_file models/validate_max_include/pumkl_150_val2.pkl
```

The same logic applies for interpretation. Simply add the `max_validation` flag:

```commandline
python scripts/scoring/compute_svc.py --max_validation 2 other_params
```

For `other_params`, refer to the main scoring script in the previous section.

#### MX-inclusion setup

For clustering, the options are very similar to MAX-inclusion. The flag `validate_max` is replaced by `validate_mx`. Note that in the case of MX validation, the number to provide must match one of the positive integers in the label file `data/pu_labels.csv`

```commandline
python scripts/clustering/main.py --validate_mx 2 --continuous_data data/continuous_data.csv --categorical_data data/categorical_data.csv --pu_labels data/pu_labels.csv --n_clusters 150 --clustering_model pumkl --clustering_file results/validate_mx_include/pumkl_150_val2.csv --model_file models/validate_mx_include/pumkl_150_val2.pkl
```

For interpretation, simply add the `mx_validation` flag:

```commandline
python scripts/scoring/compute_svc.py --mx_validation 2 other_params
```

For `other_params`, refer to the main interpretation script in the previous section.
