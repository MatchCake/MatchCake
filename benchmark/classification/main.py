import os
import sys
import psutil
import argparse
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="digits")
    parser.add_argument("--methods", type=str, nargs="+", default=["classical", "fPQC", "PQC"])
    parser.add_argument("--n_kfold_splits", type=int, default=5)
    parser.add_argument("--throw_errors", type=bool, default=False)
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--trial", type=str, default="000")
    parser.add_argument("--show_n_pts", type=int, default=512)
    parser.add_argument("--dataset_n_samples", type=int, default=None)
    parser.add_argument("--dataset_n_features", type=int, default=None)
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--simplify_qnode", type=bool, default=False)
    return parser.parse_args()


def main():
    from matplotlib import pyplot as plt
    from classification_pipeline import ClassificationPipeline

    args = parse_args()
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        print(f"Using cuda: {use_cuda}")
    except ImportError:
        use_cuda = False
    kwargs = dict(
        dataset_name=args.dataset_name,
        # dataset_name="synthetic",
        # dataset_n_samples=32,
        # dataset_n_features=2,
        # methods=[
        #     "classical",
        #     # "nif",
        #     "fPQC",
        #     "PQC",
        #     # "lightning_PQC",
        # ],
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            batch_size=args.batch_size,
            use_cuda=use_cuda,
            simplify_qnode=args.simplify_qnode
        ),
        throw_errors=args.throw_errors,
        dataset_n_samples=args.dataset_n_samples,
        dataset_n_features=args.dataset_n_features,
        use_gram_matrices=True,
    )
    save_path = os.path.join(
        args.save_dir, args.trial, f"{kwargs['dataset_name']}", f"cls.pkl"
    )
    if args.overwrite:
        pipeline = ClassificationPipeline(**kwargs)
    else:
        pipeline = ClassificationPipeline.from_pickle_or_new(save_path=save_path, **kwargs)
    figures_folder = os.path.join(os.path.dirname(save_path), "figures")
    pipeline.load_dataset()
    pipeline.preprocess_data()
    pipeline.print_summary()
    pipeline.run(results_table_path=os.path.join(figures_folder, "results_table.csv"))
    results = pipeline.get_results_table(show=True, filepath=os.path.join(figures_folder, "results_table.csv"))
    mean_results = pipeline.get_results_table(
        show=True,
        mean=True,
        filepath=os.path.join(figures_folder, "results_table_mean.csv")
    )
    pipeline.draw_mpl_kernels(show=False, filepath=os.path.join(figures_folder, "kernels.pdf"), draw_mth="single")
    plt.close("all")
    pipeline.show(n_pts=args.show_n_pts, show=True, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
    plt.close("all")


if __name__ == '__main__':
    # example of command line:
    # python benchmark/classification/main.py --dataset_name digits --methods classical fPQC PQC --trial cuda_det
    #  --batch_size 32768 --n_kfold_splits 5 --throw_errors False --show True --plot True --save_dir results
    #  --show_n_pts 512
    sys.exit(main())
