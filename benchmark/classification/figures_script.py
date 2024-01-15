import os
import pandas as pd
import matplotlib.pyplot as plt
from classification_pipeline import ClassificationPipeline


def get_table_path(dataset_name, trial="000") -> str:
    return str(os.path.join(
        os.path.dirname(__file__), "results", trial, dataset_name, "figures", "results_table_mean.csv"
    ))


if __name__ == '__main__':
    full_df = pd.DataFrame()
    columns_to_keep = [
        "Dataset",
        ClassificationPipeline.KERNEL_KEY,
        ClassificationPipeline.TEST_ACCURACY_KEY,
    ]
    datasets = ["breast_cancer", "iris"]
    for dataset in datasets:
        table_file = get_table_path(dataset, trial="000")
        df = pd.read_csv(table_file)
        df = df[df[ClassificationPipeline.KERNEL_KEY] != "classical"]
        df["Dataset"] = dataset
        full_df = pd.concat([full_df, df])
    full_table_file = get_table_path("", trial="000")
    os.makedirs(os.path.dirname(full_table_file), exist_ok=True)
    df_latex = full_df[columns_to_keep].to_latex(
        full_table_file.replace(".csv", ".tex"),
        columns=columns_to_keep,
        caption=f"Classification test accuracies for the kernels "
                f"{', '.join(full_df[ClassificationPipeline.KERNEL_KEY].unique())}"
                f" on the datasets {', '.join(datasets)}.",
        index=False
    )








