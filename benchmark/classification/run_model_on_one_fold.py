import os
import sys
from typing import Literal

import psutil
import argparse
import numpy as np
from tqdm import tqdm

try:
    import matchcake
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import matchcake
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
msim = matchcake  # Keep for compatibility with the old code


def get_memory_usage(fmt: Literal["MiB", "GiB"] = "MiB"):
    mib = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    if fmt == "MiB":
        return mib
    elif fmt == "GiB":
        return mib / 1024
    else:
        raise ValueError(f"Unknown format {fmt}")


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
        "--method", type=str,
        default="hfPQC-cpu",
        help=f"The method to be used for the classification."
             f"Available methods: {ClassificationPipeline.available_kernels}."
    )
    parser.add_argument("--n_kfold_splits", type=int, default=5)
    parser.add_argument("--fold_idx", type=int, default=2)
    parser.add_argument("--throw_errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results_dc_cluster"))
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--show_n_pts", type=int, default=512)
    parser.add_argument("--dataset_n_samples", type=int, default=None)
    parser.add_argument("--dataset_n_features", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--simplify_qnode", type=bool, default=False)
    parser.add_argument("--max_gram_size", type=int, default=np.inf)
    parser.add_argument("--kernel_size", type=int, default=28)
    return parser.parse_args()


def main():
    from matplotlib import pyplot as plt
    from classification_pipeline import ClassificationPipeline
    from utils import MPL_RC_BIG_FONT_PARAMS

    plt.rcParams.update(MPL_RC_BIG_FONT_PARAMS)
    args = parse_args()

    if "cuda" in args.method:
        matchcake.utils.cuda.is_cuda_available(throw_error=True, enable_warnings=True)
    kwargs = dict(
        dataset_name=args.dataset_name,
        methods=[args.method],
        fold_idx=args.fold_idx,
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
        args.save_dir, f"{kwargs['dataset_name']}", f"size{args.kernel_size}", ".class_pipeline"
    )
    if args.overwrite:
        pipeline = ClassificationPipeline(save_path=save_path, **kwargs)
    else:
        pipeline = ClassificationPipeline.from_dot_class_pipeline_pkl_or_new(save_path=save_path, **kwargs)
    pipeline.load_dataset()
    pipeline.preprocess_data()
    p_bar = tqdm(
        total=1,
        desc=f"Running one fold on {args.dataset_name} with "
             f"{args.method}, "
             f"fold={args.fold_idx}, "
             f"size={args.kernel_size}",
        unit="cls"
    )
    pipeline.p_bar = p_bar
    pipeline.run_fold(args.fold_idx)
    p_bar.close()
    pipeline.to_dot_class_pipeline()

    print(f"Memory usage: {get_memory_usage('MiB'):.2f} MiB, {get_memory_usage('GiB'):.2f} GiB")


if __name__ == '__main__':
    # example of command line:
    # python benchmark/classification/main.py --dataset_name digits --methods classical fPQC PQC --trial cuda_det
    #  --batch_size 32768 --n_kfold_splits 5 --throw_errors False --show True --plot True --save_dir results
    #  --show_n_pts 512
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Memory usage: {get_memory_usage('MiB'):.2f} MiB, {get_memory_usage('GiB'):.2f} GiB")
        raise e
