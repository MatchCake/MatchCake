import os
import sys

import psutil
import argparse
import numpy as np

try:
    import matchcake
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import matchcake
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
msim = matchcake  # Keep for compatibility with the old code


def parse_args():
    from classification_pipeline import ClassificationPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str,
        # default="iris",
        # default="breast_cancer",
        default="digits",
        # default="mnist",
        help=f"The dataset to be used for the classification."
             f"Available datasets: {ClassificationPipeline.available_datasets}."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        # default=["classical", "fPQC-cpu", "PQC", "wfPQC-cpu"],
        default=[
            # "classical",
            # "PQC",
            # "iPQC",
            "fPQC-cpu",
            # "fPQC-cuda",
            # "wfPQC-cuda",
            # "hfPQC-cuda",
            # "hwfPQC-cuda",
            # "ifPQC-cuda",
            # "iwfPQC-cuda",
        ],
        help=f"The methods to be used for the classification."
             f"Available methods: {ClassificationPipeline.available_kernels}."
    )
    parser.add_argument("--n_kfold_splits", type=int, default=5)
    parser.add_argument("--throw_errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--trial", type=str, default="k5")
    parser.add_argument("--show_n_pts", type=int, default=512)
    parser.add_argument("--dataset_n_samples", type=int, default=None)
    parser.add_argument("--dataset_n_features", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--simplify_qnode", type=bool, default=False)
    parser.add_argument("--max_gram_size", type=int, default=np.inf)
    parser.add_argument("--kernel_size", type=int, default=None)
    return parser.parse_args()


def main():
    from matplotlib import pyplot as plt
    from classification_pipeline import ClassificationPipeline
    from utils import MPL_RC_BIG_FONT_PARAMS

    plt.rcParams.update(MPL_RC_BIG_FONT_PARAMS)
    args = parse_args()
    if any(["cuda" in m for m in args.methods]):
        matchcake.utils.cuda.is_cuda_available(throw_error=True, enable_warnings=True)
    kwargs = dict(
        dataset_name=args.dataset_name,
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            batch_size=args.batch_size,
            simplify_qnode=args.simplify_qnode,
            size=args.kernel_size,
        ),
        throw_errors=args.throw_errors,
        dataset_n_samples=args.dataset_n_samples,
        dataset_n_features=args.dataset_n_features,
        use_gram_matrices=True,
        max_gram_size=args.max_gram_size,
    )
    save_path = os.path.join(
        args.save_dir, args.trial, f"{kwargs['dataset_name']}", f".class_pipeline"
    )
    if args.overwrite:
        pipeline = ClassificationPipeline(save_path=save_path, **kwargs)
    else:
        pipeline = ClassificationPipeline.from_dot_class_pipeline_pkl_or_new(save_path=save_path, **kwargs)
    figures_folder = os.path.join(os.path.dirname(save_path), "figures")
    pipeline.load_dataset()
    pipeline.preprocess_data()
    pipeline.print_summary()
    pipeline.run(table_path=os.path.join(figures_folder, "table.csv"))
    pipeline.to_dot_class_pipeline()
    properties = pipeline.get_properties_table(show=True, filepath=os.path.join(figures_folder, "properties.csv"))
    results = pipeline.get_results_table(show=True, filepath=os.path.join(figures_folder, "results_table.csv"))
    mean_results = pipeline.get_results_table(
        show=True,
        mean=True,
        filepath=os.path.join(figures_folder, "mean_results_table.csv")
    )
    mean_results_properties = pipeline.get_results_properties_table(
        show=True, mean=True, filepath=os.path.join(figures_folder, "mean_results_and_properties_table.csv")
    )
    pipeline.draw_mpl_kernels(show=False, filepath=os.path.join(figures_folder, "kernels.pdf"), draw_mth="single")
    plt.close("all")
    pipeline.bar_plot(
        show=args.show,
        bar_label=False,
        kernels_to_remove=["classical", ] + (
            ["wfPQC-cuda", "hwfPQC-cuda", "iwfPQC-cuda"]
            if args.dataset_name in ["digits", "mnist", "breast_cancer"]
            else []
        ),
        kernels_fmt_names={
            "classical": "Classical",
            "PQC": "PQC",
            "iPQC": r"$\otimes$PQC",
            "fPQC-cuda": "fPQC",
            "wfPQC-cuda": "wfPQC",
            "hfPQC-cuda": "hfPQC",
            "hwfPQC-cuda": "hwfPQC",
            "ifPQC-cuda": r"$\otimes$fPQC",
            "iwfPQC-cuda": r"$\otimes$wfPQC",
        },
        filepath=os.path.join(figures_folder, "bar_plot.pdf"),
    )
    plt.close("all")
    # pipeline.show(n_pts=args.show_n_pts, show=False, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
    # plt.close("all")


if __name__ == '__main__':
    # example of command line:
    # python benchmark/classification/main.py --dataset_name digits --methods classical fPQC PQC --trial cuda_det
    #  --batch_size 32768 --n_kfold_splits 5 --throw_errors False --show True --plot True --save_dir results
    #  --show_n_pts 512
    sys.exit(main())
