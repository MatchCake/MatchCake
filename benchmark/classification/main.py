from classification_pipeline import ClassificationPipeline

if __name__ == '__main__':
    pipline = ClassificationPipeline(
        dataset_name="synthetic",
        dataset_n_samples=100,
        dataset_n_features=4,
        methods=["nif"],
    )
    pipline.run()
    # pipline.print()

