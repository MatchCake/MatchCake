import time
import sys
import os
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pennylane as qml
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import umap

from kernels import ClassicalKernel, PennylaneQuantumKernel, NIFKernel
try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
from msim.ml import ClassificationVisualizer


class ClassificationPipeline:
    available_datasets = {
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "synthetic": datasets.make_classification,
    }
    available_kernels = {
        "classical": ClassicalKernel,
        "pennylane": PennylaneQuantumKernel,
        "nif": NIFKernel,
    }

    def __init__(
            self,
            dataset_name: str = "synthetic",
            methods: Optional[Union[str, List[str]]] = None,
            **kwargs
    ):
        self.classifiers = None
        self.kernels = None
        self.dataset_name = dataset_name
        self.methods = methods or list(self.available_kernels.keys())
        if isinstance(self.methods, str):
            self.methods = [self.methods]
        self.kwargs = kwargs
        self.dataset = None
        self.X, self.y = None, None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    @property
    def embedding_size(self):
        if self.X is None:
            return None
        return self.X.shape[-1]

    def load_dataset(self):
        if self.dataset_name == "synthetic":
            self.dataset = datasets.make_classification(
                n_samples=self.kwargs.get("dataset_n_samples", 100),
                n_features=self.kwargs.get("dataset_n_features", 4),
                n_classes=self.kwargs.get("dataset_n_classes", 2),
                n_clusters_per_class=self.kwargs.get("dataset_n_clusters_per_class", 1),
                n_informative=self.kwargs.get("dataset_n_informative", 2),
                n_redundant=self.kwargs.get("dataset_n_redundant", 0),
                random_state=self.kwargs.get("dataset_random_state", 0),
            )
        elif self.dataset_name == "breast_cancer":
            self.dataset = datasets.load_breast_cancer(as_frame=True)
        elif self.dataset_name == "iris":
            self.dataset = datasets.load_iris(as_frame=True)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        return self.dataset

    def preprocess_data(self):
        if isinstance(self.dataset, tuple):
            self.X, self.y = self.dataset
        elif isinstance(self.dataset, dict):
            self.X = self.dataset["data"]
            self.y = self.dataset["target"]
        elif isinstance(self.dataset, pd.DataFrame):
            self.X = self.dataset.data
            self.y = self.dataset.target
        else:
            raise ValueError(f"Unknown dataset type: {type(self.dataset)}")
        self.X = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.kwargs.get("test_size", 0.1),
            random_state=self.kwargs.get("test_split_random_state", 0),
        )
        return self.X, self.y

    def make_kernels(self):
        self.kernels = {}
        for kernel_name in self.methods:
            kernel_class = self.available_kernels[kernel_name]
            self.kernels[kernel_name] = kernel_class(
                embedding_dim=self.embedding_size,
                seed=self.kwargs.get("kernel_seed", 0),
                shots=self.kwargs.get("kernel_shots", 1),
                interface=self.kwargs.get("kernel_interface", "auto"),
                **self.kwargs.get("kernel_kwargs", {})
            )
        return self.kernels

    def fit_kernels(self):
        if self.kernels is None:
            self.make_kernels()
        for kernel_name, kernel in self.kernels.items():
            kernel.fit(self.X, self.y)
        return self.kernels

    def make_classifiers(self):
        self.classifiers = {}
        for kernel_name, kernel in self.kernels.items():
            self.classifiers[kernel_name] = svm.SVC(kernel=kernel.kernel, random_state=0)
        return self.classifiers

    def fit_classifiers(self):
        if self.classifiers is None:
            self.make_classifiers()
        for kernel_name, classifier in self.classifiers.items():
            classifier.fit(self.x_train, self.y_train)
        return self.classifiers

    def run(self):
        self.load_dataset()
        self.preprocess_data()
        self.make_kernels()
        self.fit_kernels()
        self.make_classifiers()
        self.fit_classifiers()
        return self

    def print(self):
        print(f"(N Samples, N features): {self.X.shape}")
        print(f"Classes: {set(np.unique(self.y))}, labels: {getattr(self.dataset, 'target_names', set(np.unique(self.y)))}")
        print(f"N train samples: {self.x_train.shape[0]}, N test samples: {self.x_test.shape[0]}")
        print(f"Embedding size: {self.embedding_size}")
        if "pennylane" in self.kernels:
            pennylane_kernel = self.kernels["pennylane"]
            print(f"pennylane_kernel: \n{qml.draw(pennylane_kernel.qnode)(self.X[0], self.X[-1])}\n")
        if "nif" in self.kernels:
            nif_kernel = self.kernels["nif"]
            print(f"nif_kernel: \n{qml.draw(nif_kernel.qnode)(self.X[0], self.X[-1])}\n")

    def show(
            self,
            **kwargs
    ):
        models = self.classifiers
        n_plots = len(models)
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
        axes = np.ravel(np.asarray([axes]))
        for i, (m_name, model) in enumerate(models.items()):
            fit_start_time = time.time()
            fit_end_time = time.time()
            fit_time = fit_end_time - fit_start_time
            accuracy = model.score(self.x_test, self.y_test)
            plot_start_time = time.time()
            fig, ax = ClassificationVisualizer.plot_2d_decision_boundaries(
                model=model,
                X=self.X, y=self.y,
                # reducer=decomposition.PCA(n_components=2, random_state=0),
                # reducer=umap.UMAP(n_components=2, transform_seed=0, n_jobs=max(0, psutil.cpu_count() - 2)),
                check_estimators=False,
                n_pts=1_000,
                title=f"Decision boundaries in the reduced space.",
                legend_labels=getattr(self.dataset, "target_names", None),
                # axis_name="RN",
                fig=fig, ax=axes[i],
                interpolation="nearest",
            )
            ax.set_title(f"{m_name} accuracy: {accuracy * 100:.2f}%")
            plot_end_time = time.time()
            plot_time = plot_end_time - plot_start_time
            print(f"{m_name} test accuracy: {accuracy * 100 :.4f}%, {fit_time = :.5f} [s], {plot_time = :.5f} [s]")

        if kwargs.get("show", True):
            plt.show()
        return fig, axes
