import os

import pandas as pd
import psutil
from matplotlib import pyplot as plt
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
from classification_pipeline import ClassificationPipeline


def main(**in_kwargs):
    kwargs = dict(
        # dataset_name="breast_cancer",
        dataset_name="digits",
        # dataset_name="synthetic",
        # dataset_n_samples=32,
        # dataset_n_features=2,
        methods=[
            # "classical",
            # "nif",
            "fPQC",
            "PQC",
            "PennylaneFermionicPQCKernel",
            # "lightning_PQC",
            "NeighboursFermionicPQCKernel",
        ],
        kernel_kwargs=dict(nb_workers=0),
        throw_errors=True,
    )
    kwargs.update(in_kwargs)
    save_path = os.path.join(
        os.path.dirname(__file__), "debug_results", f"{kwargs['dataset_name']}", f"cls.pkl"
    )
    pipline = ClassificationPipeline.from_pickle_or_new(
        **kwargs,
        # save_path=save_path,
        use_gram_matrices=True,
    )
    pipline.load_dataset()
    pipline.preprocess_data()
    pipline.print_summary()
    pipline.run()
    pipline.print_summary()
    figures_folder = os.path.join(os.path.dirname(save_path), "figures")
    pipline.draw_mpl_kernels(show=False, filepath=os.path.join(figures_folder, "kernels.pdf"), draw_mth="single")
    plt.close("all")
    show = kwargs.get("show", False)
    results = pipline.get_results_table(show=show, filepath=os.path.join(figures_folder, "results_table.csv"))
    if kwargs.get("plot", False):
        pipline.show(n_pts=128, show=show, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
    return results


def time_vs_n_data():
    df_data = []
    for n_data in [2**i for i in range(2, 6)]:
        results: pd.DataFrame = main(debug_data_size=n_data, show=False)
        results["n_data"] = n_data
        df_data.append(results)
    df = pd.concat(df_data)
    plt.figure()
    for kernel in df["Kernel"].unique():
        df_kernel = df[df["Kernel"] == kernel]
        plt.plot(df_kernel["n_data"], df_kernel[ClassificationPipeline.TRAIN_GRAM_COMPUTE_TIME_KEY], label=kernel)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    time_vs_n_data()
    # main(debug_data_size=32, show=True, plot=True)
