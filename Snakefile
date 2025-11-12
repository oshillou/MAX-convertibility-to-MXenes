path_to_calculated_data = "C:/Users/louoh08/OneDrive - Linköpings universitet/Documents/Postdoc project/Data/max_calculated.csv"
path_to_elemental_data = "C:/Users/louoh08/OneDrive - Linköpings universitet/Documents/Postdoc project/Data/max_elemental.csv"
path_to_synthesized_data = "C:/Users/louoh08/OneDrive - Linköpings universitet/Documents/Postdoc project/Data/synthesized-MXenes-MAX.xlsx"
python = ".\\.conda\\python.exe"

models = ["kmeans", "pumkl", "copkmeans", "pckmeans"]#, "mpckmeans"]#, "rcakmeans"]
n_positive=16
n_mxenes=10
n_clusters = [150]

wildcard_constraints:
    n_clusters="\d+",
    validation_index="\d+"

def get_validation_max_models(wildcards):
    return get_validation_models(wildcards, "max", n_positive)

def get_validation_mx_models(wildcards):
    return get_validation_models(wildcards, "mx", n_mxenes)

def get_validation_models(wildcards, compound="max", n_items=10):
    return [f"results/validate_{compound}_{wildcards.inclusion}/{wildcards.model}_{wildcards.n_clusters}_val{i}.csv" for i in range(n_items)]

def get_main_models(wildcards):
    return [f"results/{model}_{wildcards.n_clusters}.csv" for model in models]


rule all:
    input:
        expand("analysis/{model}_{n_clusters}_{result}", model=models, n_clusters=n_clusters, result=["tsne.pdf", "interpretation.txt", "scores.xlsx", "feature_importance.csv", "crossvalidated.csv"]),
        #expand("analysis/validate/{model}_{n_clusters}_validation.txt", model=models, n_clusters=n_clusters),
        #expand("analysis/validate/{model}_{n_clusters}_connectivity.pdf", model=models, n_clusters=n_clusters),
        expand("analysis/validate_max_{inclusion}/{model}_{n_clusters}_val{validation_index}_interpretation.txt", inclusion=["exclude", "include"], model=models, n_clusters=n_clusters, validation_index=range(n_positive)),
        expand("analysis/validate_mx_{inclusion}/{model}_{n_clusters}_val{validation_index}_interpretation.txt", inclusion=["exclude", "include"], model=models, n_clusters=n_clusters, validation_index=range(n_mxenes)),
        expand("analysis/validate_{phase}_{inclusion}/{model}_{n_clusters}_validation.txt", phase=["max", "mx"], inclusion=["exclude", "include"], model=models, n_clusters=n_clusters, validation_index=range(n_positive)),
        expand("analysis/pumkl_{n_clusters}_distances.csv", n_clusters=n_clusters),
        expand("analysis/{model}_{n_clusters}_eucdistances.csv", n_clusters=n_clusters, model=models),

# This rule runs fasterrisk for interpreting some clustering results
rule interprete_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/{model}_{n_clusters}.csv",
    output:
        tsne="analysis/{model}_{n_clusters}_tsne.pdf",
        interpretation="analysis/{model}_{n_clusters}_interpretation.txt",
        scores="analysis/{model}_{n_clusters}_scores.xlsx",
        importance="analysis/{model}_{n_clusters}_feature_importance.csv"
    shell:
        "{python} scripts/analysis/interprete.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --tsne_file {output.tsne} --results_file {output.interpretation} --score_file {output.scores} --feature_importance {output.importance}"

# This rule cross-validates the interpration model for the main clustering
rule crossvalidate_interpretation:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/{model}_{n_clusters}.csv",
    output:
        "analysis/{model}_{n_clusters}_crossvalidated.csv",
    shell:
        "{python} scripts/analysis/interprete_crossvalidation.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --results_file {output} --max_tests 30"


# This rule compares the main clustering (not the validation runs) by creating matrices that describe connectivities and similarities of clusterings
rule compare_clusterings:
    input:
        pu_labels="data/pu_labels.csv",
        models=get_main_models
    output:
        multiclass_ari="analysis/multiclass_ari_{n_clusters}.pdf",
        binaryclass_ari="analysis/binaryclass_ari_{n_clusters}.pdf",
        connectivity="analysis/compound_connectivity_{n_clusters}.pdf",
    shell:
        "{python} scripts/analysis/compare.py --pu_labels {input.pu_labels} --clustering_files {input.models} --multiclass_ari_plot {output.multiclass_ari} --binaryclass_ari_plot {output.binaryclass_ari} --connectivity_plot {output.connectivity}"

# This rule aggregates all the validation runs and exports some metrics for analysis
rule summarise_mx_validation:
    input:
        pu_labels="data/pu_labels.csv",
        validation=get_validation_mx_models
    output:
        text="analysis/validate_mx_{inclusion}/{model}_{n_clusters}_validation.txt",
        heatmap="analysis/validate_mx_{inclusion}/{model}_{n_clusters}_connectivity.pdf"
    shell:
        "{python} scripts/analysis/summarise_mx_validation.py --pu_labels {input.pu_labels} --clustering_files {input.validation} --validation_file {output.text} --validation_heatmap {output.heatmap}"

rule summarise_max_validation:
    input:
        pu_labels="data/pu_labels.csv",
        validation=get_validation_max_models
    output:
        text="analysis/validate_max_{inclusion}/{model}_{n_clusters}_validation.txt",
        heatmap="analysis/validate_max_{inclusion}/{model}_{n_clusters}_connectivity.pdf"
    shell:
        "{python} scripts/analysis/summarise_max_validation.py --pu_labels {input.pu_labels} --clustering_files {input.validation} --validation_file {output.text} --validation_heatmap {output.heatmap}"

# This rule runs the clustering algorithm while masking the positive sample numbe 'validation_index'
rule validate_mx_exclude_interprete:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}.csv"
    output:
        tsne="analysis/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}_tsne.pdf",
        interpretation="analysis/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}_interpretation.txt",
        scores="analysis/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}_scores.xlsx",
        importance="analysis/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}_feature_importance.csv"
    shell:
        "{python} scripts/analysis/interprete_validation.py --exclude_validation --mx_validation {wildcards.validation_index} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --tsne_file {output.tsne} --results_file {output.interpretation} --score_file {output.scores} --feature_importance {output.importance}"


rule validate_mx_exclude_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}.csv",
        model="models/validate_mx_exclude/{model}_{n_clusters}_val{validation_index}.csv"
    shell:
        "{python} scripts/clustering/main.py --exclude_validation --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --validate_mx {wildcards.validation_index} --model_file {output.model}"


rule validate_max_exclude_interprete:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/validate_max_exclude/{model}_{n_clusters}_val{validation_index}.csv"
    output:
        tsne="analysis/validate_max_exclude/{model}_{n_clusters}_val{validation_index}_tsne.pdf",
        interpretation="analysis/validate_max_exclude/{model}_{n_clusters}_val{validation_index}_interpretation.txt",
        scores="analysis/validate_max_exclude/{model}_{n_clusters}_val{validation_index}_scores.xlsx",
        importance="analysis/validate_max_exclude/{model}_{n_clusters}_val{validation_index}_feature_importance.csv"
    shell:
        "{python} scripts/analysis/interprete_validation.py --exclude_validation --max_validation {wildcards.validation_index} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --tsne_file {output.tsne} --results_file {output.interpretation} --score_file {output.scores} --feature_importance {output.importance}"


rule validate_max_exclude_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/validate_max_exclude/{model}_{n_clusters}_val{validation_index}.csv",
        model="models/validate_max_exclude/{model}_{n_clusters}_val{validation_index}.csv"
    shell:
        "{python} scripts/clustering/main.py --exclude_validation --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --validate_max {wildcards.validation_index} --model_file {output.model}"

rule validate_mx_include_interprete:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/validate_mx_include/{model}_{n_clusters}_val{validation_index}.csv"
    output:
        tsne="analysis/validate_mx_include/{model}_{n_clusters}_val{validation_index}_tsne.pdf",
        interpretation="analysis/validate_mx_include/{model}_{n_clusters}_val{validation_index}_interpretation.txt",
        scores="analysis/validate_mx_include/{model}_{n_clusters}_val{validation_index}_scores.xlsx",
        importance="analysis/validate_mx_include/{model}_{n_clusters}_val{validation_index}_feature_importance.csv"
    shell:
        "{python} scripts/analysis/interprete_validation.py --mx_validation {wildcards.validation_index} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --tsne_file {output.tsne} --results_file {output.interpretation} --score_file {output.scores} --feature_importance {output.importance}"


rule validate_mx_include_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/validate_mx_include/{model}_{n_clusters}_val{validation_index}.csv",
        model="models/validate_mx_include/{model}_{n_clusters}_val{validation_index}.csv"
    shell:
        "{python} scripts/clustering/main.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --validate_mx {wildcards.validation_index} --model_file {output.model}"

rule validate_max_include_interprete:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv",
        clustering="results/validate_max_include/{model}_{n_clusters}_val{validation_index}.csv"
    output:
        tsne="analysis/validate_max_include/{model}_{n_clusters}_val{validation_index}_tsne.pdf",
        interpretation="analysis/validate_max_include/{model}_{n_clusters}_val{validation_index}_interpretation.txt",
        scores="analysis/validate_max_include/{model}_{n_clusters}_val{validation_index}_scores.xlsx",
        importance="analysis/validate_max_include/{model}_{n_clusters}_val{validation_index}_feature_importance.csv"
    shell:
        "{python} scripts/analysis/interprete_validation.py --max_validation {wildcards.validation_index} --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --clustering_file {input.clustering} --tsne_file {output.tsne} --results_file {output.interpretation} --score_file {output.scores} --feature_importance {output.importance}"


rule validate_max_include_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/validate_max_include/{model}_{n_clusters}_val{validation_index}.csv",
        model="models/validate_max_include/{model}_{n_clusters}_val{validation_index}.csv"
    shell:
        "{python} scripts/clustering/main.py --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --validate_max {wildcards.validation_index} --model_file {output.model}"

# This rule (specificaly for pumkl) computes the distance between each sample and the biggest positive cluster centroid
rule pumkl_distance:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        model="models/pumkl_{n_clusters}.pkl",
        clustering="results/pumkl_{n_clusters}.csv"
    output:
        "analysis/pumkl_{n_clusters}_distances.csv"
    shell:
        "{python} scripts/clustering/compute_distance_pumkl.py --continuous_data {input.continuous} --categorical_data {input.categorical} --model_file {input.model} --clustering_file {input.clustering} --output_file {output} "

rule other_distance:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        clustering="results/{model}_{n_clusters}.csv"
    output:
        "analysis/{model}_{n_clusters}_eucdistances.csv"
    shell:
        "{python} scripts/clustering/compute_distance.py --continuous_data {input.continuous} --categorical_data {input.categorical} --clustering_file {input.clustering} --output_file {output} "

# This rule performs the final global clustering (note the seed for reproducibility)
rule perform_clustering:
    input:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    output:
        clustering="results/{model}_{n_clusters}.csv",
        model="models/{model}_{n_clusters}.pkl"
    shell:
        "{python} scripts/clustering/main.py --seed 0 --continuous_data {input.continuous} --categorical_data {input.categorical} --pu_labels {input.pu_labels} --n_clusters {wildcards.n_clusters} --clustering_model {wildcards.model} --clustering_file {output.clustering} --model_file {output.model}"



# This rule creates the dataset (scaled features from original source)
rule make_dataset:
    input:
        calculated=path_to_calculated_data,
        elemental=path_to_elemental_data,
        synthesized=path_to_synthesized_data,
    output:
        continuous="data/continuous_data.csv",
        categorical="data/categorical_data.csv",
        pu_labels="data/pu_labels.csv"
    shell:
        "{python} scripts/data/main.py --max_calculated \"{input.calculated}\" --max_elemental \"{input.elemental}\" --max_synthesized \"{input.synthesized}\" --output_continuous {output.continuous} --output_categorical {output.categorical} --output_targets {output.pu_labels}"