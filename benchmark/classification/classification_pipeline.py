import time
import sys
import os
from copy import deepcopy
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
from tqdm import tqdm

from kernels import (
    ClassicalKernel,
    MPennylaneQuantumKernel,
    CPennylaneQuantumKernel,
    NIFKernel,
    FermionicPQCKernel,
    PQCKernel,
)
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
        "m_pennylane": MPennylaneQuantumKernel,
        "c_pennylane": CPennylaneQuantumKernel,
        "nif": NIFKernel,
        "fPQC": FermionicPQCKernel,
        "PQC": PQCKernel,
    }

    def __init__(
            self,
            dataset_name: str = "synthetic",
            methods: Optional[Union[str, List[str]]] = None,
            **kwargs
    ):
        self.classifiers = {}
        self.kernels = {}
        self.dataset_name = dataset_name
        self.methods = methods or list(self.available_kernels.keys())
        if isinstance(self.methods, str):
            self.methods = [self.methods]
        self.kwargs = kwargs
        self.dataset = None
        self.X, self.y = None, None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.fit_times = {}
        self.fit_kernels_times = {}
        self.plot_times = {}
        self.test_accuracies = {}
        self.train_accuracies = {}
        self.save_path = kwargs.get("save_path", None)
        self._has_run = False
        self.train_gram_matrices = {}
        self.test_gram_matrices = {}

    @property
    def n_features(self):
        if self.X is None:
            return None
        return self.X.shape[-1]

    @property
    def has_run(self):
        return self._has_run

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
        self.X = MinMaxScaler(feature_range=self.kwargs.get("feature_range", (0, 1))).fit_transform(self.X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.kwargs.get("test_size", 0.1),
            random_state=self.kwargs.get("test_split_random_state", 0),
        )
        return self.X, self.y

    def make_kernels(self):
        self.kernels = {}
        for kernel_name in self.methods:
            try:
                kernel_class = self.available_kernels[kernel_name]
                self.kernels[kernel_name] = kernel_class(
                    seed=self.kwargs.get("kernel_seed", 0),
                    **self.kwargs.get("kernel_kwargs", {})
                )
            except Exception as e:
                print(f"Failed to make kernel {kernel_name}: {e}")
        return self.kernels

    def fit_kernels(self):
        if self.kernels is None:
            self.make_kernels()
        to_remove = []
        for kernel_name, kernel in self.kernels.items():
            try:
                start_time = time.perf_counter()
                kernel.fit(self.x_train, self.y_train)
                self.fit_kernels_times[kernel_name] = time.perf_counter() - start_time
            except Exception as e:
                if self.kwargs.get("throw_errors", False):
                    raise e
                print(f"Failed to fit kernel {kernel_name}: {e}. \n Removing it from the list of kernels.")
                to_remove.append(kernel_name)
        for kernel_name in to_remove:
            self.kernels.pop(kernel_name)
        return self.kernels

    def compute_gram_matrices(self):
        if self.kernels is None:
            self.make_kernels()
        to_remove = []
        for kernel_name, kernel in self.kernels.items():
            try:
                start_time = time.perf_counter()
                self.train_gram_matrices[kernel_name] = kernel.pairwise_distances(self.x_train, self.x_train.T)
                self.test_gram_matrices[kernel_name] = kernel.pairwise_distances(self.x_test, self.x_train.T)
                self.fit_kernels_times[kernel_name] = time.perf_counter() - start_time
            except Exception as e:
                if self.kwargs.get("throw_errors", False):
                    raise e
                print(f"Failed to compute gram matrix for kernel {kernel_name}: {e}. \n Removing it from the list of kernels.")
                to_remove.append(kernel_name)
        for kernel_name in to_remove:
            self.kernels.pop(kernel_name)
        return self.kernels

    def make_classifiers(self):
        self.classifiers = {}
        for kernel_name, kernel in self.kernels.items():
            self.classifiers[kernel_name] = svm.SVC(kernel=kernel.pairwise_distances, random_state=0)
        return self.classifiers

    def fit_classifiers(self):
        if self.classifiers is None:
            self.make_classifiers()
        p_bar = tqdm(self.classifiers.items(), desc="Fitting classifiers", unit="cls")
        to_remove = []
        for kernel_name, classifier in self.classifiers.items():
            try:
                start_time = time.perf_counter()
                classifier.fit(self.train_gram_matrices[kernel_name], self.y_train)
                self.fit_times[kernel_name] = time.perf_counter() - start_time
            except Exception as e:
                if self.kwargs.get("throw_errors", False):
                    raise e
                print(f"Failed to fit classifier {kernel_name}: {e}. \n Removing it from the list of classifiers.")
                to_remove.append(kernel_name)
                p_bar.update()
                continue
            self.test_accuracies[kernel_name] = classifier.score(self.test_gram_matrices[kernel_name], self.y_test)
            self.train_accuracies[kernel_name] = classifier.score(self.train_gram_matrices[kernel_name], self.y_train)
            p_bar.update()
            p_bar.set_postfix({
                f"{kernel_name} fit time": f"{self.fit_times.get(kernel_name, np.NaN):.2f} [s]",
                f"{kernel_name} test accuracy": f"{self.test_accuracies.get(kernel_name, np.NaN) * 100:.2f} %",
                f"{kernel_name} train accuracy": f"{self.train_accuracies.get(kernel_name, np.NaN) * 100:.2f} %",
            })
        p_bar.close()
        for kernel_name in to_remove:
            self.classifiers.pop(kernel_name)
            self.kernels.pop(kernel_name)
        return self.classifiers

    def run(self):
        if self._has_run:
            return self
        self.load_dataset()
        self.preprocess_data()
        self.make_kernels()
        self.fit_kernels()
        self.compute_gram_matrices()
        self.make_classifiers()
        self.fit_classifiers()
        self._has_run = True
        self.to_pickle()
        return self

    def print(self):
        print(f"(N Samples, N features): {self.X.shape}")
        print(f"Classes: {set(np.unique(self.y))}, labels: {getattr(self.dataset, 'target_names', set(np.unique(self.y)))}")
        print(f"N train samples: {self.x_train.shape[0]}, N test samples: {self.x_test.shape[0]}")

        def _print_times(m_name):
            print(
                f"{m_name} test accuracy: {self.test_accuracies.get(m_name, np.NaN) * 100 :.4f}%, "
                f"train accuracy: {self.train_accuracies.get(m_name, np.NaN) * 100 :.4f}%, "
                f"fit time: {self.fit_times.get(m_name, np.NaN):.5f} [s], "
                f"plot time: {self.plot_times.get(m_name, np.NaN):.5f} [s]"
            )
        
        for m_name, kernel in self.kernels.items():
            if not hasattr(kernel, "draw"):
                continue
            kernel.draw(name=m_name, logging_func=print)
        
        for m_name in self.kernels:
            _print_times(m_name)

    def draw_mpl_kernels_single(self, **kwargs):
        kernels = kwargs.pop("kernels", list(self.kernels.keys()))
        filepath = kwargs.pop("filepath", None)
        show = kwargs.pop("show", False)
        figs, axes = [], []
        for i, kernel_name in enumerate(kernels):
            if filepath is not None:
                filepath = filepath.replace(".", f"_{kernel_name}.")
                kwargs["filepath"] = filepath
            fig, ax = self.kernels[kernel_name].draw_mpl(**kwargs)
            ax.set_title(kernel_name)
            figs.append(fig)
            axes.append(ax)
        if show:
            plt.show()
        return figs, axes

    def draw_mpl_kernels(self, fig: Optional[plt.Figure] = None, axes: Optional[np.ndarray] = None, **kwargs):
        draw_mth = kwargs.pop("draw_mth", "single")
        if draw_mth == "single":
            return self.draw_mpl_kernels_single(**deepcopy(kwargs))

        kernels = kwargs.pop("kernels", list(self.kernels.keys()))
        n_plots = len(kernels)
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_plots, 1, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
        axes = np.ravel(np.asarray([axes]))
        assert len(axes) >= n_plots, f"The number of axes ({len(axes)}) is less than the number of kernels ({kernels})."
        filepath = kwargs.get("filepath", None)
        for i, kernel_name in enumerate(kernels):
            self.kernels[kernel_name].draw_mpl(fig=fig, ax=axes[i], **kwargs)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)
        if kwargs.get("show", False):
            plt.show()
        return fig, axes

    def show(
            self,
            fig: Optional[plt.Figure] = None,
            axes: Optional[np.ndarray] = None,
            **kwargs
    ):
        kwargs.setdefault("check_estimators", False)
        kwargs.setdefault("n_pts", 512)
        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("title", f"Decision boundaries in the reduced space.")
        
        show = kwargs.pop("show", True)
        models = kwargs.pop("models", self.classifiers)
        n_plots = len(models)
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
        axes = np.ravel(np.asarray([axes]))
        assert len(axes) >= n_plots, f"The number of axes ({len(axes)}) is less than the number of models ({models})."
        p_bar = tqdm(models.items(), desc="Plotting decision boundaries", unit="model")
        for i, (m_name, model) in enumerate(models.items()):
            plot_start_time = time.perf_counter()
            fig, ax = ClassificationVisualizer.plot_2d_decision_boundaries(
                model=model,
                X=self.X, y=self.y,
                # reducer=decomposition.PCA(n_components=2, random_state=0),
                # reducer=umap.UMAP(n_components=2, transform_seed=0, n_jobs=max(0, psutil.cpu_count() - 2)),
                legend_labels=getattr(self.dataset, "target_names", None),
                fig=fig, ax=axes[i],
                show=False,
                **kwargs
            )
            self.plot_times[m_name] = time.perf_counter() - plot_start_time
            ax.set_title(f"{m_name} accuracy: {self.test_accuracies.get(m_name, np.NaN) * 100:.2f}%")
            p_bar.update()
            p_bar.set_postfix({
                f"{m_name} plot time": f"{self.plot_times.get(m_name, np.NaN):.2f} [s]",
                f"{m_name} fit time": f"{self.fit_times.get(m_name, np.NaN):.2f} [s]",
                f"{m_name} test accuracy": f"{self.test_accuracies.get(m_name, np.NaN) * 100:.2f} %",
                f"{m_name} train accuracy": f"{self.train_accuracies.get(m_name, np.NaN) * 100:.2f} %",
            })
        p_bar.close()

        filepath = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)

        if show:
            plt.show()
        return fig, axes

    def plot(self, *args, **kwargs):
        return self.show(*args, **kwargs)

    def to_pickle(self):
        if self.save_path is not None:
            import pickle
            save_path = self.save_path
            if not save_path.endswith(".pkl"):
                save_path += ".pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, pickle_path: str) -> "ClassificationPipeline":
        import pickle
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_pickle_or_new(cls, pickle_path: Optional[str] = None, **kwargs) -> "ClassificationPipeline":
        save_path = kwargs.get("save_path", None)
        if pickle_path is None:
            pickle_path = save_path
        if pickle_path is not None and os.path.exists(pickle_path):
            return cls.from_pickle(pickle_path)
        else:
            return cls(**kwargs)
