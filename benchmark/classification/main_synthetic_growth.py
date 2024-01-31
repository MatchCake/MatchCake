import argparse
import os
import sys
import numpy as np
import psutil
from matplotlib import pyplot as plt

try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


def parse_args():
    from classification_pipeline import ClassificationPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", type=str, nargs="+", default=[
            # "classical",
            # "fPQC-cpu",
            # "fPQC",
            "fPQC-cuda",
            "PQC",
            "wfPQC-cuda",
            # "wfPQC-cpu",
        ],
        help=f"The methods to be used for the classification."
             f"Example: --methods fPQC PQC."
             f"Available methods: {ClassificationPipeline.available_kernels}."
    )
    parser.add_argument(
        "--n_kfold_splits", type=int, default=5,
        help="The number of kfold splits to be used for the classification."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096,
        help="The batch size to be used for the classification."
    )
    parser.add_argument(
        "--save_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results_sg/1024"),
        help="The directory where the results will be saved."
    )
    parser.add_argument(
        "--n_features_list", type=int, nargs="+", default=None,
        help=f"The list of number of features to be used for the classification."
             f"Example: --n_features_list 2 4 8 16."
    )
    parser.add_argument("--throw_errors", type=bool, default=True)
    return parser.parse_args()


def main():
    from classification_pipeline import SyntheticGrowthPipeline, ClassificationPipeline

    args = parse_args()
    if any(["cuda" in m for m in args.methods]):
        msim.utils.cuda.is_cuda_available(throw_error=True, enable_warnings=True)

    classification_pipeline_kwargs = dict(
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            batch_size=args.batch_size,
        ),
        throw_errors=args.throw_errors,
    )
    pipeline = SyntheticGrowthPipeline(
        n_features_list=args.n_features_list,
        save_dir=args.save_dir,
        classification_pipeline_kwargs=classification_pipeline_kwargs
    )
    pipeline.run()
    pipeline.get_results_table(show=True)

    x_keys = ["kernel_size", "n_features"]
    for x_key in x_keys:
        y_keys = [
            ClassificationPipeline.FIT_TIME_KEY,
            "kernel_n_ops",
            # ClassificationPipeline.TEST_ACCURACY_KEY,
            "kernel_depth",
        ] + [x for x in x_keys if x != x_key]
        n_rows = int(np.sqrt(len(y_keys)))
        n_cols = int(np.ceil(len(y_keys) / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.ravel(np.asarray([axes]))
        for i, y_key in enumerate(y_keys):
            pipeline.plot_results(x_axis_key=x_key, y_axis_key=y_key, fig=fig, ax=axes[i], show=False)
            axes[i].set_title(y_key)
        plt.tight_layout()
        if args.save_dir is not None:
            fig.savefig(os.path.join(args.save_dir, f"results_{x_key}.pdf"), bbox_inches="tight", dpi=900)
        plt.show()
        plt.close("all")


if __name__ == '__main__':
    sys.exit(main())
