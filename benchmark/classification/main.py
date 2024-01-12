import os
import psutil
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from classification_pipeline import ClassificationPipeline
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        print(f"Using cuda: {use_cuda}")
    except ImportError:
        use_cuda = False
    trial = "cuda_det"
    kwargs = dict(
        dataset_name="digits",
        # dataset_name="synthetic",
        # dataset_n_samples=32,
        # dataset_n_features=2,
        methods=[
            "classical",
            # "nif",
            "fPQC",
            "PQC",
            # "lightning_PQC",
        ],
        kernel_kwargs=dict(nb_workers=0, batch_size=int(2**15), use_cuda=use_cuda),
        throw_errors=False,
    )
    save_path = os.path.join(
        os.path.dirname(__file__), "results", trial, f"{kwargs['dataset_name']}", f"cls.pkl"
    )
    pipline = ClassificationPipeline.from_pickle_or_new(
        **kwargs,
        save_path=save_path,
        use_gram_matrices=True,
    )
    figures_folder = os.path.join(os.path.dirname(save_path), "figures")
    pipline.load_dataset()
    pipline.preprocess_data()
    pipline.print_summary()
    pipline.run(results_table_path=os.path.join(figures_folder, "results_table.csv"))
    results = pipline.get_results_table(show=True, filepath=os.path.join(figures_folder, "results_table.csv"))
    mean_results = pipline.get_results_table(
        show=True,
        mean=True,
        filepath=os.path.join(figures_folder, "results_table_mean.csv")
    )
    pipline.draw_mpl_kernels(show=False, filepath=os.path.join(figures_folder, "kernels.pdf"), draw_mth="single")
    plt.close("all")
    pipline.show(n_pts=512, show=True, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
    plt.close("all")
