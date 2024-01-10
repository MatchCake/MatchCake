import time
import sys
import os
from copy import deepcopy
from collections import defaultdict
from typing import Optional, Union, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pennylane as qml
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
import functools
import umap
from tqdm import tqdm
import pythonbasictools as pbt

from kernels import (
    ClassicalKernel,
    MPennylaneQuantumKernel,
    CPennylaneQuantumKernel,
    NIFKernel,
    FermionicPQCKernel,
    PQCKernel,
    LightningPQCKernel,
    PennylaneFermionicPQCKernel,
    NeighboursFermionicPQCKernel,
)

try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
from msim.ml import ClassificationVisualizer
from msim.ml.ml_kernel import MLKernel
import warnings


def save_on_exit(_method=None, *, save_func_name="to_pickle", save_args=(), **save_kwargs):
    """
    Decorator for a method that saves the object on exit.

    :param _method: The method to decorate.
    :type _method: Callable
    :param save_func_name: The name of the method that saves the object.
    :type save_func_name: str
    :param save_args: The arguments of the save method.
    :type save_args: tuple
    :param save_kwargs: The keyword arguments of the save method.
    :return: The decorated method.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Failed to run method {method.__name__}: {e}", RuntimeWarning)
                raise e
            finally:
                self = args[0]
                if not hasattr(self, save_func_name):
                    warnings.warn(
                        f"The object {self.__class__.__name__} does not have a save method named {save_func_name}.",
                        RuntimeWarning
                    )
                try:
                    getattr(self, save_func_name)(*save_args, **save_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to save object {self.__class__.__name__}: {e} with {save_func_name} method",
                        RuntimeWarning
                    )
                    # raise e

        wrapper.__name__ = method.__name__ + "@save_on_exit"
        return wrapper

    if _method is None:
        return decorator
    else:
        return decorator(_method)


def get_gram_predictor(cls, kernel, x_train, **kwargs):
    def predictor(x_test):
        return cls.predict(kernel.pairwise_distances(x_test, x_train, **kwargs))
    return predictor


class KPredictorContainer:
    def __init__(self, name: str = ""):
        self.name = name
        self.container = defaultdict(dict)

    def get(self, key, inner_key, default_value=None):
        return self.container.get(key, {}).get(inner_key, default_value)

    def set(self, key, inner_key, value):
        self.container[key][inner_key] = value

    def get_inner(self, key):
        return self.container[key]

    def get_outer(self, inner_key, default_value=None):
        outer_dict = {key: default_value for key in self.container}
        for key, inner_dict in self.container.items():
            for _inner_key, value in inner_dict.items():
                if _inner_key == inner_key:
                    outer_dict[key] = value
        return outer_dict

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            return self.set(key[0], key[1], value)
        else:
            raise ValueError("Key in __setitem__ must be a tuple")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.get(item[0], item[1])
        else:
            return self.get_inner(item)

    def items(self, default_value=None):
        for key in self.keys():
            yield key, self.get(*key, default_value=default_value)

    def keys(self):
        for key, inner_dict in self.container.items():
            for inner_key in inner_dict:
                yield key, inner_key

    def to_dataframe(
            self,
            *,
            outer_column: str = "outer",
            inner_column: str = "inner",
            value_column: str = "value",
            default_value=None,
    ) -> pd.DataFrame:
        all_keys = list(self.keys())
        all_outer_keys = list(set(k[0] for k in all_keys))
        all_inner_keys = list(set(k[1] for k in all_keys))
        df_dict = {
            outer_column: [],
            inner_column: [],
            value_column: [],
        }
        for ok in all_outer_keys:
            for ik in all_inner_keys:
                df_dict[outer_column].append(ok)
                df_dict[inner_column].append(ik)
                df_dict[value_column].append(self.get(ok, ik, default_value))
        return pd.DataFrame(df_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},outer_keys={list(self.container.keys())})"


class ClassificationPipeline:
    available_datasets = {
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "synthetic": datasets.make_classification,
        "mnist": fetch_openml,
        "digits": datasets.load_digits,
    }
    available_kernels = {
        "classical": ClassicalKernel,
        "m_pennylane": MPennylaneQuantumKernel,
        "c_pennylane": CPennylaneQuantumKernel,
        "nif": NIFKernel,
        "fPQC": FermionicPQCKernel,
        "PQC": PQCKernel,
        "lightning_PQC": LightningPQCKernel,
        "PennylaneFermionicPQCKernel": PennylaneFermionicPQCKernel,
        "nfPQC": NeighboursFermionicPQCKernel,
    }
    UNPICKLABLE_ATTRIBUTES = ["dataset", ]

    KERNEL_KEY = "Kernel"
    CLASSIFIER_KEY = "Classifier"
    FIT_TIME_KEY = "Fit Time [s]"
    TEST_ACCURACY_KEY = "Test Accuracy [-]"
    TRAIN_ACCURACY_KEY = "Train Accuracy [-]"
    PLOT_TIME_KEY = "Plot Time [s]"
    TRAIN_GRAM_COMPUTE_TIME_KEY = "Train Gram Compute Time [s]"
    TEST_GRAM_COMPUTE_TIME_KEY = "Test Gram Compute Time [s]"
    FIT_KERNEL_TIME_KEY = "Fit Kernel Time [s]"
    FOLD_IDX_KEY = "Fold Idx [-]"
    
    def __init__(
            self,
            dataset_name: str = "synthetic",
            methods: Optional[Union[str, List[str]]] = None,
            **kwargs
    ):
        self.classifiers = KPredictorContainer("classifiers")
        self.kernels = KPredictorContainer("kernels")
        self.dataset_name = dataset_name
        self.methods = methods or list(self.available_kernels.keys())
        if isinstance(self.methods, str):
            self.methods = [self.methods]
        self.kwargs = kwargs
        self.dataset = None
        self.X, self.y = None, None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.fit_times = KPredictorContainer(self.FIT_TIME_KEY)
        self.fit_kernels_times = KPredictorContainer(self.FIT_KERNEL_TIME_KEY)
        self.plot_times = KPredictorContainer(self.PLOT_TIME_KEY)
        self.test_accuracies = KPredictorContainer(self.TEST_ACCURACY_KEY)
        self.train_accuracies = KPredictorContainer(self.TRAIN_ACCURACY_KEY)
        self.save_path = kwargs.get("save_path", None)
        self.train_gram_matrices = KPredictorContainer("train_gram_matrices")
        self.test_gram_matrices = KPredictorContainer("test_gram_matrices")
        self.train_gram_compute_times = KPredictorContainer(self.TRAIN_GRAM_COMPUTE_TIME_KEY)
        self.test_gram_compute_times = KPredictorContainer(self.TEST_GRAM_COMPUTE_TIME_KEY)
        self.use_gram_matrices = self.kwargs.get("use_gram_matrices", False)
        self._db_y_preds = KPredictorContainer("_db_y_preds")
        self._debug_data_size = self.kwargs.get("debug_data_size", None)
        self._n_class = self.kwargs.get("n_class", None)
        self.dataframe = pd.DataFrame(columns=[
            self.KERNEL_KEY, self.CLASSIFIER_KEY, self.FIT_TIME_KEY, self.TEST_ACCURACY_KEY, self.TRAIN_ACCURACY_KEY,
            self.PLOT_TIME_KEY, self.TRAIN_GRAM_COMPUTE_TIME_KEY, self.TEST_GRAM_COMPUTE_TIME_KEY,
            self.FIT_KERNEL_TIME_KEY,
        ])
        self._n_kfold_splits = self.kwargs.get("n_kfold_splits", 5)
        self._kfold_random_state = self.kwargs.get("kfold_random_state", 0)
        self._kfold_shuffle = self.kwargs.get("kfold_shuffle", True)

    @property
    def n_features(self):
        if self.X is None:
            return None
        return self.X.shape[-1]
    
    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.load_dataset()
        self.preprocess_data()

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
        elif self.dataset_name == "mnist":
            self.dataset = fetch_openml(
                "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
            )
        elif self.dataset_name == "digits":
            n_class = self._n_class or 10
            self.dataset = self.available_datasets[self.dataset_name](as_frame=True, n_class=n_class)
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
        if self._debug_data_size is not None:
            self.X = self.X[:self._debug_data_size]
            self.y = self.y[:self._debug_data_size]
        self.X = MinMaxScaler(feature_range=self.kwargs.get("feature_range", (0, 1))).fit_transform(self.X)
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        #     self.X, self.y,
        #     test_size=self.kwargs.get("test_size", 0.2),
        #     random_state=self.kwargs.get("test_split_random_state", 0),
        # )
        return self.X, self.y

    def get_train_test_fold(self, fold_idx: int):
        if self._n_kfold_splits <= 1:
            return train_test_split(
                self.X, self.y,
                test_size=self.kwargs.get("test_size", 0.2),
                random_state=self.kwargs.get("test_split_random_state", 0),
            )
        from sklearn.model_selection import KFold
        kf = KFold(
            n_splits=self._n_kfold_splits,
            shuffle=self._kfold_shuffle,
            random_state=self._kfold_random_state,
        )
        train_indexes, test_indexes = list(kf.split(self.X, self.y))[fold_idx]
        x_train, x_test = self.X[train_indexes], self.X[test_indexes]
        y_train, y_test = self.y[train_indexes], self.y[test_indexes]
        return x_train, x_test, y_train, y_test
    
    @save_on_exit
    def make_kernels(self, fold_idx: int = 0):
        for kernel_name in self.methods:
            if self.kernels.get(kernel_name, fold_idx, None) is not None:
                continue
            try:
                kernel_class = self.available_kernels[kernel_name]
                self.kernels[kernel_name, fold_idx] = kernel_class(
                    seed=self.kwargs.get("kernel_seed", 0),
                    **self.kwargs.get("kernel_kwargs", {})
                )
            except Exception as e:
                print(f"Failed to make kernel {kernel_name}: {e}")
        return self.kernels
    
    @save_on_exit
    def fit_kernels(self, fold_idx: int = 0):
        if not self.kernels:
            self.make_kernels()
        to_remove = []
        x_train, x_test, y_train, y_test = self.get_train_test_fold(fold_idx)
        for kernel_name, kernel in self.kernels.get_outer(fold_idx).items():
            if kernel.is_fitted:
                continue
            try:
                start_time = time.perf_counter()
                kernel.fit(x_train, y_train)
                elapsed_time = time.perf_counter() - start_time
                self.fit_kernels_times[kernel_name, fold_idx] = elapsed_time
                # self.dataframe[self.dataframe[self.KERNEL_KEY == kernel_name]][self.FIT_KERNEL_TIME_KEY] = elapsed_time
            except Exception as e:
                if self.kwargs.get("throw_errors", False):
                    raise e
                print(f"Failed to fit kernel {kernel_name}: {e}. \n Removing it from the list of kernels.")
                to_remove.append(kernel_name)
        for kernel_name in to_remove:
            self.kernels[kernel_name].pop(fold_idx)
        return self.kernels
    
    @save_on_exit
    def compute_train_gram_matrices(self, fold_idx: int = 0):
        if not self.kernels:
            self.make_kernels()
        x_train, x_test, y_train, y_test = self.get_train_test_fold(fold_idx)
        to_remove = []
        verbose = self.kwargs.get("verbose_gram", True)
        throw_errors = self.kwargs.get("throw_errors", False)
        for kernel_name, kernel in self.kernels.get_outer(fold_idx).items():
            if self.train_gram_matrices.get(kernel_name, fold_idx, None) is not None:
                continue
            try:
                start_time = time.perf_counter()
                self.train_gram_matrices[kernel_name, fold_idx] = kernel.compute_gram_matrix(
                    x_train, verbose=verbose, throw_errors=throw_errors
                )
                self.train_gram_compute_times[kernel_name, fold_idx] = time.perf_counter() - start_time
            except Exception as e:
                if throw_errors:
                    raise e
                warnings.warn(
                    f"Failed to compute gram matrix for kernel {kernel_name}: {e}. "
                    f"\n Removing it from the list of kernels.",
                    RuntimeWarning
                )
                to_remove.append(kernel_name)
        for kernel_name in to_remove:
            self.kernels[kernel_name].pop(fold_idx)
        return self.kernels
    
    @save_on_exit
    def compute_test_gram_matrices(self, fold_idx: int = 0):
        if not self.kernels:
            self.make_kernels()
        x_train, x_test, y_train, y_test = self.get_train_test_fold(fold_idx)
        p_bar = tqdm(self.kernels.get_outer(fold_idx).items(), desc=f"Compute Gram Matrices k={fold_idx}", unit="cls")
        to_remove = []
        verbose = self.kwargs.get("verbose_gram", True)
        throw_errors = self.kwargs.get("throw_errors", False)
        for kernel_name, kernel in p_bar:
            if self.test_gram_matrices.get(kernel_name, fold_idx, None) is not None:
                continue
            try:
                start_time = time.perf_counter()
                self.test_gram_matrices[kernel_name, fold_idx] = kernel.pairwise_distances(
                    x_test, x_train, verbose=verbose, throw_errors=throw_errors
                )
                self.test_gram_compute_times[kernel_name, fold_idx] = time.perf_counter() - start_time
            except Exception as e:
                if throw_errors:
                    raise e
                print(f"Failed to compute gram matrix for kernel {kernel_name}: {e}. "
                      f"\n Removing it from the list of kernels.")
                to_remove.append(kernel_name)
                p_bar.update()
        p_bar.close()
        for kernel_name in to_remove:
            self.kernels[kernel_name].pop(fold_idx)
        return self.kernels
    
    @save_on_exit
    def compute_gram_matrices(self, fold_idx: int = 0):
        self.compute_train_gram_matrices(fold_idx)
        self.compute_test_gram_matrices(fold_idx)
        return self.kernels
    
    @save_on_exit
    def make_classifiers(self, fold_idx: int = 0):
        cache_size = self.kwargs.get("kernel_cache_size", 10_000)
        for kernel_name, kernel in self.kernels.get_outer(fold_idx).items():
            if self.classifiers.get(kernel_name, fold_idx, None) is not None:
                continue
            if self.use_gram_matrices:
                svc_kernel = "precomputed"
            else:
                svc_kernel = kernel.pairwise_distances
            self.classifiers[kernel_name, fold_idx] = svm.SVC(kernel=svc_kernel, random_state=0, cache_size=cache_size)
        return self.classifiers
    
    @save_on_exit
    def fit_classifiers(self, fold_idx: int = 0):
        if self.classifiers is None:
            self.make_classifiers()
        p_bar = tqdm(self.classifiers.get_outer(fold_idx).items(), desc=f"Fitting classifiers k={fold_idx}", unit="cls")
        x_train, x_test, y_train, y_test = self.get_train_test_fold(fold_idx)
        to_remove = []
        for kernel_name, classifier in self.classifiers.get_outer(fold_idx).items():
            is_fitted = True
            try:
                check_is_fitted(classifier)
            except Exception as e:
                is_fitted = False
            if (self.fit_times.get(kernel_name, fold_idx, None) is not None) and is_fitted:
                continue
            try:
                start_time = time.perf_counter()
                if classifier.kernel == "precomputed":
                    classifier.fit(self.train_gram_matrices[kernel_name, fold_idx], y_train)
                else:
                    classifier.fit(x_train, y_train)
                self.fit_times[kernel_name, fold_idx] = time.perf_counter() - start_time
            except Exception as e:
                if self.kwargs.get("throw_errors", False):
                    raise e
                print(f"Failed to fit classifier {kernel_name}: {e}. \n Removing it from the list of classifiers.")
                to_remove.append(kernel_name)
                p_bar.update()
                continue
            if classifier.kernel == "precomputed":
                train_inputs, test_inputs = (
                    self.train_gram_matrices[kernel_name, fold_idx],
                    self.test_gram_matrices[kernel_name, fold_idx]
                )
            else:
                train_inputs, test_inputs = x_train, x_test
            self.train_accuracies[kernel_name, fold_idx] = classifier.score(train_inputs, y_train)
            self.test_accuracies[kernel_name, fold_idx] = classifier.score(test_inputs, y_test)
            p_bar.update()
            p_bar.set_postfix({
                f"{kernel_name} fit time": f"{self.fit_times.get(kernel_name, fold_idx, np.NaN):.2f} [s]",
                f"{kernel_name} test accuracy":
                    f"{self.test_accuracies.get(kernel_name, fold_idx, np.NaN) * 100:.2f} %",
                f"{kernel_name} train accuracy":
                    f"{self.train_accuracies.get(kernel_name, fold_idx, np.NaN) * 100:.2f} %",
            })
        p_bar.close()
        for kernel_name in to_remove:
            self.classifiers[kernel_name].pop(fold_idx)
            self.kernels[kernel_name].pop(fold_idx)
        return self.classifiers

    @pbt.decorators.log_func(logging_func=print)
    @save_on_exit
    def run(self, **kwargs):
        results_table_path = kwargs.pop("results_table_path", None)
        show_results_table = kwargs.pop("show_results_table", False)
        mean_results_table = kwargs.pop("mean_results_table", False)
        self.load_dataset()
        self.preprocess_data()
        for fold_idx in range(self._n_kfold_splits):
            self.run_fold(fold_idx)
            if results_table_path is not None:
                self.get_results_table(
                    mean=mean_results_table,
                    show=show_results_table,
                    filepath=results_table_path
                )
        self.to_pickle()
        return self

    @save_on_exit
    def run_fold(self, fold_idx: int):
        self.make_kernels(fold_idx)
        self.fit_kernels(fold_idx)
        if self.use_gram_matrices:
            self.compute_gram_matrices(fold_idx)
        self.make_classifiers(fold_idx)
        self.fit_classifiers(fold_idx)
    
    def print_summary(self):
        print(f"(N Samples, N features): {self.X.shape}")
        print(f"Classes: {set(np.unique(self.y))}, labels: {getattr(self.dataset, 'target_names', set(np.unique(self.y)))}")
        # print(f"N train samples: {self.X.shape[0]}, N test samples: {self.x_test.shape[0]}")

        def _print_times(m_name):
            print(
                f"{m_name} test accuracy: {self.test_accuracies.get(*m_name, np.NaN) * 100 :.4f}%, "
                f"train accuracy: {self.train_accuracies.get(*m_name, np.NaN) * 100 :.4f}%, "
                f"fit time: {self.fit_times.get(*m_name, np.NaN):.5f} [s], "
                f"fit kernel time: {self.fit_kernels_times.get(*m_name, np.NaN):.5f} [s], "
                f"train gram compute time: {self.train_gram_compute_times.get(*m_name, np.NaN):.5f} [s], "
                f"test gram compute time: {self.test_gram_compute_times.get(*m_name, np.NaN):.5f} [s], "
                f"plot time: {self.plot_times.get(*m_name, np.NaN):.5f} [s]"
            )
        try:
            for m_name, kernel in self.kernels.items():
                if not hasattr(kernel, "draw"):
                    continue
                kernel.draw(name=m_name, logging_func=print)
        except Exception as e:
            warnings.warn(f"Failed to draw kernels: {e}", RuntimeWarning)
        
        for m_name in self.kernels.keys():
            _print_times(m_name)

    def draw_mpl_kernels_single(self, **kwargs):
        fold_idx = kwargs.get("fold_idx", 0)
        kernels = kwargs.pop("kernels", list(self.kernels.get_outer(fold_idx).keys()))
        kernels = [kernel for kernel in kernels if hasattr(self.kernels[kernel, fold_idx], "draw_mpl")]
        filepath = kwargs.pop("filepath", None)
        show = kwargs.pop("show", False)
        figs, axes = [], []
        for i, kernel_name in enumerate(kernels):
            if filepath is not None:
                filepath = filepath.replace(".", f"_{kernel_name}.")
                kwargs["filepath"] = filepath
            fig, ax = self.kernels[kernel_name, fold_idx].draw_mpl(**kwargs)
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
        kernels = [kernel for kernel in kernels if hasattr(self.kernels[kernel], "draw_mpl")]
        n_plots = len(kernels)
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_plots, 1, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
        axes = np.ravel(np.asarray([axes]))
        assert len(axes) >= n_plots, f"The number of axes ({len(axes)}) is less than the number of kernels ({kernels})."
        filepath = kwargs.get("filepath", None)
        for i, kernel_name in enumerate(kernels):
            _fig, _axes = self.kernels[kernel_name].draw_mpl(fig=fig, ax=axes[i], **kwargs)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)
        if kwargs.get("show", False):
            plt.show()
        if kwargs.get("close", True):
            plt.close(fig)
        return fig, axes

    @save_on_exit
    def show(
            self,
            fig: Optional[plt.Figure] = None,
            axes: Optional[np.ndarray] = None,
            *,
            fold_idx: int = 0,
            **kwargs
    ):
        kwargs.setdefault("check_estimators", False)
        kwargs.setdefault("n_pts", 512)
        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("title", f"Decision boundaries in the reduced space.")
        
        _show = kwargs.pop("show", True)
        models = kwargs.pop("models", self.classifiers.get_outer(fold_idx))
        n_plots = len(models)
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))
        if fig is None or axes is None:
            fig, axes = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
        axes = np.ravel(np.asarray([axes]))
        assert len(axes) >= n_plots, f"The number of axes ({len(axes)}) is less than the number of models ({models})."
        p_bar = tqdm(models.items(), desc="Plotting decision boundaries", unit="model")
        viz = ClassificationVisualizer(x=self.X, **kwargs)
        for i, (m_name, model) in enumerate(models.items()):
            if self.use_gram_matrices:
                predict_func = get_gram_predictor(model, self.kernels[m_name, fold_idx], self.x_train)
            else:
                predict_func = getattr(model, "predict", None)
            y_pred = self._db_y_preds.get(m_name, fold_idx, None)
            plot_start_time = time.perf_counter()
            fig, ax, y_pred = viz.plot_2d_decision_boundaries(
                model=model,
                y=self.y,
                y_pred=y_pred,
                predict_func=predict_func,
                legend_labels=getattr(self.dataset, "target_names", None),
                fig=fig, ax=axes[i],
                show=False,
                **kwargs
            )
            self.plot_times[m_name, fold_idx] = time.perf_counter() - plot_start_time
            self._db_y_preds[m_name, fold_idx] = y_pred
            ax.set_title(f"{m_name} accuracy: {self.test_accuracies.get(m_name, fold_idx, np.NaN) * 100:.2f}%")
            p_bar.update()
            p_bar.set_postfix({
                f"{m_name} plot time": f"{self.plot_times.get(m_name, fold_idx, np.NaN):.2f} [s]",
                f"{m_name} fit time": f"{self.fit_times.get(m_name, fold_idx, np.NaN):.2f} [s]",
                f"{m_name} test accuracy": f"{self.test_accuracies.get(m_name, fold_idx, np.NaN) * 100:.2f} %",
                f"{m_name} train accuracy": f"{self.train_accuracies.get(m_name, fold_idx, np.NaN) * 100:.2f} %",
            })
        p_bar.close()

        filepath = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)
        if _show:
            plt.show()
        return fig, axes

    def plot(self, *args, **kwargs):
        return self.show(*args, **kwargs)

    @pbt.decorators.log_func
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
    def from_pickle(cls, pickle_path: str, new_attrs: Optional[dict] = None) -> "ClassificationPipeline":
        import pickle
        with open(pickle_path, "rb") as f:
            loaded_obj = pickle.load(f)
        if new_attrs is not None:
            loaded_obj.__dict__.update(new_attrs)
        return loaded_obj
    
    @classmethod
    def from_pickle_or_new(cls, pickle_path: Optional[str] = None, **kwargs) -> "ClassificationPipeline":
        save_path = kwargs.get("save_path", None)
        if pickle_path is None:
            pickle_path = save_path
        if pickle_path is not None and os.path.exists(pickle_path):
            try:
                return cls.from_pickle(pickle_path, new_attrs=kwargs)
            except EOFError:
                warnings.warn(
                    f"Failed to load object from pickle file {pickle_path}. Encountered EOFError. "
                    f"Loading a new object instead.",
                    RuntimeWarning
                )

        return cls(**kwargs)

    def to_npz(self):
        if self.save_path is not None:
            save_path = self.save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **self.__getstate__())

    def get_results_table(self, **kwargs):
        containers = [
            self.train_accuracies,
            self.test_accuracies,
            self.fit_times,
            self.fit_kernels_times,
            self.train_gram_compute_times,
            self.test_gram_compute_times,
            self.plot_times,
        ]
        numerics_columns = [c.name for c in containers]
        df_list = [
            c.to_dataframe(
                outer_column=self.KERNEL_KEY,
                inner_column=self.FOLD_IDX_KEY,
                value_column=c.name,
                default_value=np.NaN
            )
            for c in containers
        ]
        df = functools.reduce(
            lambda df1, df2: pd.merge(df1, df2, on=[self.KERNEL_KEY, self.FOLD_IDX_KEY], how="outer"),
            df_list
        )
        sort = kwargs.get("sort", False)
        is_sorted = False
        mean_df = kwargs.get("mean", False)
        mean_on = kwargs.get("mean_on", self.KERNEL_KEY)
        drop_on_mean = kwargs.get("drop_on_mean", self.FOLD_IDX_KEY)
        if mean_df:
            df_mean = df.groupby(by=mean_on, as_index=False).mean().drop(drop_on_mean, axis=1)
            # if sort:
            #     df_mean = df_mean.sort_values(by=self.TEST_ACCURACY_KEY, ascending=False)
            #     is_sorted = True
            df_std = df.groupby(by=mean_on, as_index=False).std().drop(drop_on_mean, axis=1)
            map_func = lambda x: f"{x:.4f}"
            df_wo_numerics = df_mean[numerics_columns].map(map_func) + u"\u00B1" + df_std[numerics_columns].map(map_func)
            df = pd.concat([df_mean[[mean_on]], df_wo_numerics], axis=1)

        if sort:
            df = df.sort_values(by=self.TEST_ACCURACY_KEY, ascending=False)
        filepath: Optional[str] = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath)
        if kwargs.get("show", False):
            print(df.to_markdown())
        return df
