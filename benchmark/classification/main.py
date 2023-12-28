import os
import psutil
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from classification_pipeline import ClassificationPipeline

    kwargs = dict(
        dataset_name="digits",
        # dataset_name="synthetic",
        # dataset_n_samples=32,
        # dataset_n_features=2,
        methods=[
            # "classical",
            # "nif",
            "fPQC",
            # "PQC",
            # "lightning_PQC",
        ],
        kernel_kwargs=dict(nb_workers=0),
        throw_errors=True,
    )
    save_path = os.path.join(
        os.path.dirname(__file__), "results", f"{kwargs['dataset_name']}", f"cls.pkl"
    )
    pipline = ClassificationPipeline.from_pickle_or_new(
        **kwargs,
        save_path=save_path,
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
    pipline.show(n_pts=512, show=True, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
