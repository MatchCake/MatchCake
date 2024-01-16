import os
import pandas as pd
import matplotlib.pyplot as plt
from classification_pipeline import ClassificationPipeline


def get_results_table_path(dataset_name, trial="000") -> str:
    return str(os.path.join(
        os.path.dirname(__file__), "results", trial, dataset_name, "figures", "results_table_mean.csv"
    ))


def make_results_table():
    full_df = pd.DataFrame()
    columns_to_keep = [
        "Dataset",
        ClassificationPipeline.KERNEL_KEY,
        ClassificationPipeline.TEST_ACCURACY_KEY,
        ClassificationPipeline.TEST_F1_KEY,
    ]
    datasets_fmt_names = {
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "Olivetti_faces": "Olivetti faces",
        "SignMNIST": "SignMNIST",
        "mnist": "MNIST",
        "Fashion-MNIST": "Fashion-MNIST",
    }
    for dataset, dataset_fmt_name in datasets_fmt_names.items():
        table_file = get_results_table_path(dataset, trial="000")
        if not os.path.isfile(table_file):
            continue
        df = pd.read_csv(table_file)
        df = df[df[ClassificationPipeline.KERNEL_KEY] != "classical"]
        df["Dataset"] = dataset_fmt_name
        full_df = pd.concat([full_df, df])

    full_table_file = get_results_table_path("", trial="000")
    os.makedirs(os.path.dirname(full_table_file), exist_ok=True)
    df_latex = full_df[columns_to_keep].to_latex(
        full_table_file.replace(".csv", ".tex"),
        columns=columns_to_keep,
        caption=(
            f"Classification test accuracies for the kernels "
            f"{', '.join(full_df[ClassificationPipeline.KERNEL_KEY].unique())}"
            f" on the datasets {', '.join(list(datasets_fmt_names.values()))}.",
            f"Classification test accuracies.",
        ),
        index=False
    )


if __name__ == '__main__':
    make_results_table()

