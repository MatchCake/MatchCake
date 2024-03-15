import os
import pandas as pd
import numpy as np
from classification_pipeline import ClassificationPipeline


def get_table_path(dataset_name, trial="cls_k5") -> str:
    return str(os.path.join(
        os.path.dirname(__file__), "results", trial, dataset_name, "figures", "mean_results_and_properties_table.csv"
    ))


def pm_string_to_floats(pm_string, pm_char="±"):
    return np.asarray([float(v) for v in pm_string.split(pm_char)])


def floats_to_pm_string(floats, pm_char="±", formatter="{:.2f}"):
    return pm_char.join([formatter.format(v) for v in floats])


def make_results_table(trial="cls_k5"):
    full_df = pd.DataFrame()
    columns_to_keep = [
        "Dataset",
        ClassificationPipeline.KERNEL_KEY,
        "kernel_depth",
        "kernel_size",
        ClassificationPipeline.TEST_ACCURACY_KEY,
        ClassificationPipeline.TEST_F1_KEY,
    ]
    columns_fmt_names = {
        "Dataset": "Dataset",
        ClassificationPipeline.KERNEL_KEY: "Kernel",
        "kernel_depth": "D",
        "kernel_size": "N",
        ClassificationPipeline.TEST_ACCURACY_KEY: r"Accuracy [\%]",
        ClassificationPipeline.TEST_F1_KEY: r"F1 score [\%]",
    }
    integer_columns = ["kernel_depth", "kernel_size"]
    datasets_fmt_names = {
        "iris": "Iris",
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "Olivetti_faces": "Olivetti faces",
        "SignMNIST": "SignMNIST",
        "mnist": "MNIST",
        "Fashion-MNIST": "Fashion-MNIST",
    }
    DATASET_KEY = "Dataset"
    for dataset, dataset_fmt_name in datasets_fmt_names.items():
        table_file = get_table_path(dataset, trial=trial)
        if not os.path.isfile(table_file):
            continue
        df = pd.read_csv(table_file)
        df = df[df[ClassificationPipeline.KERNEL_KEY] != "classical"]
        df[DATASET_KEY] = dataset_fmt_name
        full_df = pd.concat([full_df, df])

    all_kernels = full_df[ClassificationPipeline.KERNEL_KEY].unique()
    gpu_kernels = [m for m in all_kernels if "cuda" in m]
    cpu_kernels = [m for m in all_kernels if m not in gpu_kernels]
    all_kernels_set = cpu_kernels + [m for m in gpu_kernels if m.replace("-cuda", "-cpu") not in cpu_kernels]
    kernels_to_keep = ["PQC", "iPQC", "fPQC", "ifPQC", "hfPQC"]
    all_kernels_set = [
        m for m in all_kernels_set
        if any([
            k.replace("-cuda", "").replace("-cpu", "") == m.replace("-cuda", "").replace("-cpu", "")
            for k in kernels_to_keep
        ])
    ]
    kernels_to_fmt_names = {
        "PQC": "PQC",
        "fPQC": "fPQC",
        "ifPQC": r"$\otimes$fPQC",
        "hfPQC": "hfPQC",
        "wfPQC": "wfPQC",
        "hwfPQC": "hwfPQC",
        "iwfPQC": r"$\otimes$wfPQC",
    }
    # filter the kernels
    full_df = full_df[full_df[ClassificationPipeline.KERNEL_KEY].isin(all_kernels_set)]
    # use pm_string_to_floats to convert the accuracies and f1 to floats, multiply by 100,
    # find the max by dataset and convert back to string and bold the max
    for key in [
        ClassificationPipeline.TEST_ACCURACY_KEY,
        ClassificationPipeline.TRAIN_ACCURACY_KEY,
        ClassificationPipeline.TEST_F1_KEY,
    ]:
        full_df[key] = full_df[key].apply(pm_string_to_floats)
        full_df[key] *= 100
        full_df[key] = full_df[key].apply(floats_to_pm_string)

    # cast the integer_columns to int
    for c in integer_columns:
        full_df[c] = full_df[c].astype(int)

    # rename the kernels
    full_df[ClassificationPipeline.KERNEL_KEY] = full_df[ClassificationPipeline.KERNEL_KEY].apply(
        lambda x: kernels_to_fmt_names.get(x, x)
    )
    # rename the columns
    full_df = full_df.rename(columns=columns_fmt_names)
    columns_to_keep = [columns_fmt_names[c] for c in columns_to_keep]

    full_table_file = get_table_path("", trial=trial)
    os.makedirs(os.path.dirname(full_table_file), exist_ok=True)
    df_latex = full_df[columns_to_keep].to_latex(
        full_table_file.replace(".csv", ".tex"),
        columns=columns_to_keep,
        caption=(
            f"Classification test accuracies for the kernels "
            f"{', '.join(full_df[ClassificationPipeline.KERNEL_KEY].unique())}"
            f" on the datasets {', '.join(full_df[DATASET_KEY].unique())}."
            f" The columns D and N denote the depth and the number of qubits of the kernels, respectively.",
            f"Classification test accuracies.",
        ),
        index=False
    )


if __name__ == '__main__':
    make_results_table(trial="k5")
