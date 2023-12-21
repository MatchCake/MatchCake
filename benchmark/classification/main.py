import os

from classification_pipeline import ClassificationPipeline

if __name__ == '__main__':
    kwargs = dict(
        # dataset_name="breast_cancer",
        dataset_name="synthetic",
        dataset_n_samples=32,
        dataset_n_features=2,
        methods=[
            "classical",
            # "nif",
            # "fPQC",
            # "PQC",
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
    )
    pipline.load_dataset()
    pipline.preprocess_data()
    pipline.print()
    pipline.run()
    pipline.print()
    figures_folder = os.path.join(os.path.dirname(save_path), "figures")
    pipline.draw_mpl_kernels(show=True, filepath=os.path.join(figures_folder, "kernels.pdf"), draw_mth="single")
    pipline.show(n_points=128, show=True, filepath=os.path.join(figures_folder, "decision_boundaries.pdf"))
