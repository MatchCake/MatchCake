import argparse
import os
import sys
import pandas as pd
import psutil
from matplotlib import pyplot as plt
try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", type=str, nargs="+", default=[
            # "classical",
            "fPQC",
            "PQC"
        ],
        help="The methods to be used for the classification."
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
        "--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results_sg"),
        help="The directory where the results will be saved."
    )
    parser.add_argument(
        "--n_features_list", type=str, nargs="+", default=None,
        help="The list of number of features to be used for the classification."
    )
    parser.add_argument(
        "--use_cuda", type=bool, default=True,
        help="Whether to use CUDA or not."
    )
    parser.add_argument("--throw_errors", type=bool, default=True)
    return parser.parse_args()


def main():
    from classification_pipeline import SyntheticGrowthPipeline

    args = parse_args()
    use_cuda = args.use_cuda and msim.utils.cuda.is_cuda_available(enable_warnings=args.use_cuda)

    classification_pipeline_kwargs = dict(
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            use_cuda=use_cuda,
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
    pipeline.plot_results(show=True)


if __name__ == '__main__':
    sys.exit(main())
