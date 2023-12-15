from classification_pipeline import ClassificationPipeline

if __name__ == '__main__':
    pipline = ClassificationPipeline(
        dataset_name="synthetic",
        dataset_n_samples=32,
        dataset_n_features=16,
        methods=[
            # "classical",
            "nif",
        ],
        kernel_kwargs=dict(nb_workers=0),
    )
    pipline.load_dataset()
    pipline.preprocess_data()
    pipline.print()
    pipline.run()
    pipline.print()
    pipline.show(n_points=128, show=False)
