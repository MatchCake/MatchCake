import argparse
import os
import sys
import pandas as pd
import psutil
from matplotlib import pyplot as plt
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["classical", "fPQC", "PQC"],
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
    return parser.parse_args()


def main():
    from classification_pipeline import SyntheticGrowthPipeline

    args = parse_args()
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        print(f"Using cuda: {use_cuda}")
    except ImportError:
        use_cuda = False

    classification_pipeline_kwargs = dict(
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            use_cuda=use_cuda,
            batch_size=args.batch_size,
        ),
    )
    pipeline = SyntheticGrowthPipeline(
        n_features_list=args.n_features_list,
        save_dir=args.save_dir,
        classification_pipeline_kwargs=classification_pipeline_kwargs
    )
    pipeline.run()


if __name__ == '__main__':
    sys.exit(main())
