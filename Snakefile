python = ".\\.conda\\python.exe"

models = ["kmeans", "pumkl", "copkmeans", "pckmeans"]#, "mpckmeans"]#, "rcakmeans"]
n_positive=16
n_mxenes=10
n_clusters = [150]
years_list = [2011,2012,2013,2014,2016,2017,2018,2019,2020,2024]

wildcard_constraints:
    n_clusters="\d+",
    validation_index="\d+",
    leaveout_mode="(max_include|max_exclude|mx_include|mx_exclude)",
    scoring_model="(fasterrisk|forest)",
    model="(pumkl|pckmeans|kmeans|copkmeans)"

def get_clustering_leaveout_params(wildcards):
    phase, inclusion = wildcards.leaveout_mode.split("_")
    params = []
    if phase == "max":
        params += ["--validate_max"]
    else:
        params += ["--validate_mx"]
    params += [str(wildcards.validation_index)]
    if inclusion=="exclude":
        params += ["--exclude_validation"]
    return " ".join(params)



rule all:
    input:
        expand("results/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores.xlsx",
            scoring_filename=[config["scoring_filename"]],
            model_filename=[config["clustering_filename"]],
            scoring_model=[config["scoring_model"]],
            model=[config["clustering_model"]],
            n_clusters=[config["n_clusters"]]
        ),
        expand("validation/timeline/scoring/{year}/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores.xlsx",
            year=years_list,
            scoring_filename=[config["scoring_filename"]],
            model_filename=[config["clustering_filename"]],
            scoring_model=[config["scoring_model"]],
            model=[config["clustering_model"]],
            n_clusters=[config["n_clusters"]]
        ),
        expand("validation/max_{inclusion}/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores_val{validation_index}.xlsx",
            inclusion=["include","exclude"],
            scoring_filename=[config["scoring_filename"]],
            model_filename=[config["clustering_filename"]],
            scoring_model=[config["scoring_model"]],
            model=[config["clustering_model"]],
            n_clusters=[config["n_clusters"]],
            validation_index = range(n_positive)
        ),
        expand("validation/mx_{inclusion}/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores_val{validation_index}.xlsx",
            inclusion=["include","exclude"],
            scoring_filename=[config["scoring_filename"]],
            model_filename=[config["clustering_filename"]],
            scoring_model=[config["scoring_model"]],
            model=[config["clustering_model"]],
            n_clusters=[config["n_clusters"]],
            validation_index=range(n_mxenes)
        ),
        expand("results/visualisation/{model_filename}/{model}_{n_clusters}.pdf",
           model_filename=[config["clustering_filename"]],
           model=[config["clustering_model"]],
           n_clusters=[config["n_clusters"]]
        ),

rule validation_scoring_fasterrisk:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="validation/{leaveout_mode}/clustering/{model_filename}/{model}_{n_clusters}_clusters_val{validation_index}.csv"
    output:
        interpretation="validation/{leaveout_mode}/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_interpretation_val{validation_index}.txt",
        scores="validation/{leaveout_mode}/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores_val{validation_index}.xlsx",
    shell:
        "{python} scripts/scoring/compute_{wildcards.scoring_model}.py --exclude_validation --mx_validation {wildcards.validation_index} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --results_file {output.interpretation} --score_file {output.scores} "+config["scoring_params"]


rule timeline_scoring:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels_{year}.csv",
        clustering="validation/timeline/clustering/{year}/{model_filename}/{model}_{n_clusters}_clusters.csv"
    output:
        interpretation="validation/timeline/scoring/{year}/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_interpretation.txt",
        scores="validation/timeline/scoring/{year}/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores.xlsx",
    shell:
        "{python} scripts/scoring/compute_{wildcards.scoring_model}.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --results_file {output.interpretation} --score_file {output.scores} "+config["scoring_params"]

rule main_scoring:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/clustering/{model_filename}/{model}_{n_clusters}_clusters.csv"
    output:
        interpretation="results/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_interpretation.txt",
        scores="results/scoring/{scoring_filename}/{model_filename}/{scoring_model}_{model}_{n_clusters}_scores.xlsx",
    shell:
        "{python} scripts/scoring/compute_{wildcards.scoring_model}.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --results_file {output.interpretation} --score_file {output.scores} "+config["scoring_params"]


# performs the leave-"one"-out cross validation for the clustering model
# Masked MAX or MX compounds may be included or excluded from the training set of a clustering algorithm
# In both cases, their label is masked
rule validate_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="validation/{leaveout_mode}/clustering/{model_filename}/{model}_{n_clusters}_clusters_val{validation_index}.csv",
        model="validation/{leaveout_mode}/clustering/{model_filename}/{model}_{n_clusters}_weights_val{validation_index}.pkl",
        distance="validation/{leaveout_mode}/clustering/{model_filename}/{model}_{n_clusters}_distances_val{validation_index}.csv"
    params:
        get_clustering_leaveout_params
    shell:
        "{python} scripts/clustering/main.py {params} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --model_file {output.model} --distance_file {output.distance}"

# Creates a tsne of the main clustering results
rule visualise_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical_data="data/categorical_data.csv",
        clustering="results/clustering/{model_filename}/{model}_{n_clusters}_clusters.csv",
    output:
        "results/visualisation/{model_filename}/{model}_{n_clusters}.pdf"
    shell:
        "{python} scripts/visualisation/compute_tsne.py --continuous_data {input.continuous} --categorical_data {input.categorical_data} --clustering_file {input.clustering} --output_file {output}"

# This rule performs the main clustering on the dataset while using a different set of constraints for timeline
# #validation
rule perform_timeline_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels_{year}.csv"
    output:
        clustering="validation/timeline/clustering/{year}/{model_filename}/{model}_{n_clusters}_clusters.csv",
        model="validation/timeline/clustering/{year}/{model_filename}/{model}_{n_clusters}_weights.pkl",
        distance="validation/timeline/clustering/{year}/{model_filename}/{model}_{n_clusters}_distances.csv"
    shell:
        "{python} scripts/clustering/main.py --seed 0 --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --model_file {output.model} --distance_file {output.distance}"

# This rule performs the final global clustering (note the seed for reproducibility)
rule perform_main_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/clustering/{model_filename}/{model}_{n_clusters}_clusters.csv",
        model="results/clustering/{model_filename}/{model}_{n_clusters}_weights.pkl",
        distance="results/clustering/{model_filename}/{model}_{n_clusters}_distances.csv"
    shell:
        "{python} scripts/clustering/main.py --seed 0 --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --model_file {output.model} --distance_file {output.distance}"

# For the timeline validation, this rule masks out convertible MAX phases that were discovered after a given year
rule filter_targets_by_year:
    input:
        pu_labels="data/pu_labels.csv",
        year_csv=config["path_to_label_years"],
    output:
        "data/pu_labels_{year}.csv"
    shell:
        "{python} scripts/data/filter_labels_by_year.py --pu_labels {input.pu_labels} --year_csv {input.year_csv} --year_cutoff {wildcards.year} --output_labels {output}"

# This rule creates the dataset (scaled features from original source)
rule make_dataset:
    input:
        calculated=config["path_to_calculated_data"],
        elemental=config["path_to_elemental_data"],
        synthesized=config["path_to_synthesized_data"],
    output:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    shell:
        "{python} scripts/data/main.py --max_calculated \"{input.calculated}\" --max_elemental \"{input.elemental}\" --max_synthesized \"{input.synthesized}\" --output_continuous {output.continuous} --output_categorical {output.categorical} --output_targets {output.pu_labels}"