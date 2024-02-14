import time
import sys
import os
from copy import deepcopy
from collections import defaultdict
from typing import Optional, Union, List, Callable
from functools import partial
import argparse
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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
    CudaFermionicPQCKernel,
    CpuFermionicPQCKernel,
    WideFermionicPQCKernel,
    CpuWideFermionicPQCKernel,
    CudaWideFermionicPQCKernel,
    FastCudaFermionicPQCKernel,
    FastCudaWideFermionicPQCKernel,
    SwapCudaFermionicPQCKernel,
    SwapCudaWideFermionicPQCKernel,
    IdentityCudaFermionicPQCKernel,
    IdentityCudaWideFermionicPQCKernel,
    HadamardCudaWideFermionicPQCKernel,
    HadamardCudaFermionicPQCKernel,
)

try:
    import matchcake
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import matchcake
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
from matchcake.ml import ClassificationVisualizer
from matchcake.ml.ml_kernel import MLKernel, FixedSizeSVC
import warnings
from classification_pipeline import ClassificationPipeline


class DatasetComplexityPipeline:
    DATASET_MAX_SIZE_MAP = {
        "digits": 64,
        "iris": 4,
        "breast_cancer": 30,
    }
    DEFAULT_SIZE_LISTS = {
        "digits": np.arange(2, DATASET_MAX_SIZE_MAP["digits"], 2).tolist(),
        "iris": np.arange(2, DATASET_MAX_SIZE_MAP["iris"], 2).tolist(),
        "breast_cancer": np.arange(2, DATASET_MAX_SIZE_MAP["breast_cancer"], 2).tolist(),
    }
    MTH_MAX_SIZE_MAP = {
        "fPQC-cuda": np.inf,
        "fPQC-cpu": np.inf,
        "PQC": 30,
    }

    def __init__(
            self,
            kernel_size_list: Optional[List[int]] = None,
            n_samples: Optional[int] = None,
            dataset_name: str = "digits",
            classification_pipeline_kwargs: Optional[dict] = None,
            save_dir: Optional[str] = None,
    ):
        if kernel_size_list is None:
            kernel_size_list = self.DEFAULT_SIZE_LISTS.get(dataset_name, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        self.kernel_size_list = kernel_size_list
        self.n_samples = n_samples
        self.dataset_name = dataset_name
        self.classification_pipeline_kwargs = classification_pipeline_kwargs or {}
        self.classification_pipelines = {}
        self._results_table = None
        self.save_dir = save_dir

    @property
    def results_table(self):
        if self._results_table is None:
            self.get_results_table()
        return self._results_table

    @results_table.setter
    def results_table(self, value):
        self._results_table = value

    def filter_methods_by_max_size(self, methods: List[str], size: int) -> List[str]:
        return [m for m in methods if size <= self.MTH_MAX_SIZE_MAP.get(m, np.inf)]

    def run(self, **kwargs):
        save_dir = kwargs.get("save_dir", self.save_dir)
        n_mth = len(self.classification_pipeline_kwargs.get("methods", ClassificationPipeline.available_kernels))
        n_kfold_splits = self.classification_pipeline_kwargs.get(
            "n_kfold_splits", ClassificationPipeline.DEFAULT_N_KFOLD_SPLITS
        )
        n_itr = len(self.kernel_size_list) * n_mth * n_kfold_splits
        p_bar = tqdm(
            np.arange(n_itr),
            desc=f"Complexity run on {self.dataset_name}",
        )
        kwargs["p_bar"] = p_bar
        kwargs["close_p_bar"] = False
        for size in self.kernel_size_list:
            cp_kwargs = deepcopy(self.classification_pipeline_kwargs)
            cp_kwargs["dataset_name"] = self.dataset_name
            cp_kwargs["kernel_kwargs"] = cp_kwargs.get("kernel_kwargs", {})
            cp_kwargs["kernel_kwargs"].update(dict(size=size))
            methods = self.filter_methods_by_max_size(cp_kwargs.get("methods", []), size)
            cp_kwargs["methods"] = methods
            cp_kwargs["dataset_n_samples"] = self.n_samples
            if save_dir is not None:
                cp_kwargs["save_path"] = os.path.join(
                    save_dir, f"{self.dataset_name}_size{size}.pkl"
                )
            self.classification_pipelines[size] = ClassificationPipeline.from_pickle_or_new(
                pickle_path=cp_kwargs.get("save_path", None), **cp_kwargs
            ).run(**kwargs)
        p_bar.close()
        return self

    def get_results_table(self, **kwargs):
        df_list = []
        show = kwargs.pop("show", False)
        filepath: Optional[str] = kwargs.pop("filepath", None)
        for n_features, pipeline in self.classification_pipelines.items():
            df_list.append(pipeline.get_results_properties_table(**kwargs))
        df = pd.concat(df_list)
        self.results_table = df

        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath)
        if show:
            print(df.to_markdown())
        return df

    def plot_results(self, **kwargs):
        x_axis_key = kwargs.get("x_axis_key", "kernel_size")
        y_axis_key = kwargs.get("y_axis_key", ClassificationPipeline.FIT_TIME_KEY)
        df = self.results_table
        fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (kernel_name, kernel_df) in enumerate(df.groupby(ClassificationPipeline.KERNEL_KEY)):
            x_sorted_unique = np.sort(kernel_df[x_axis_key].unique())
            y_series = kernel_df.groupby(x_axis_key)[y_axis_key]
            y_mean = y_series.mean().values
            y_std = y_series.std().values
            ax.plot(x_sorted_unique, y_mean, label=kernel_name, color=colors[i])
            if y_series.count().min() > 1:
                ax.fill_between(x_sorted_unique, y_mean - y_std, y_mean + y_std, alpha=0.2, color=colors[i])
        ax.set_xlabel(x_axis_key)
        ax.set_ylabel(y_axis_key)
        ax.legend()
        if kwargs.get("show", False):
            plt.show()
        return fig, ax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str,
        default="digits",
        help=f"The name of the dataset to be used for the classification. "
             f"Available datasets: {DatasetComplexityPipeline.DATASET_MAX_SIZE_MAP.keys()}."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=[
            # "fPQC-cpu",
            "fPQC-cuda",
            # "hfPQC-cuda",
            # "ifPQC-cuda",
            # "PQC",
        ],
        help=f"The methods to be used for the classification."
             f"Example: --methods fPQC PQC."
             f"Available methods: {ClassificationPipeline.available_kernels}."
    )
    parser.add_argument(
        "--n_kfold_splits", type=int, default=5,
        help="The number of kfold splits to be used for the classification."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16384,  # 32768
        help="The batch size to be used for the classification."
    )
    parser.add_argument(
        "--save_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results_dc"),
        help="The directory where the results will be saved."
    )
    parser.add_argument(
        "--kernel_size_list", type=int, nargs="+", default=None,
        help=f"The list of number of qubits to be used for the classification."
             f"Example: --kernel_size_list 2 4 8 16."
    )
    parser.add_argument("--throw_errors", type=bool, default=True)
    parser.add_argument("--n_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(["cuda" in m for m in args.methods]):
        matchcake.utils.cuda.is_cuda_available(throw_error=True, enable_warnings=True)

    classification_pipeline_kwargs = dict(
        methods=args.methods,
        n_kfold_splits=args.n_kfold_splits,
        kernel_kwargs=dict(
            nb_workers=0,
            batch_size=args.batch_size,
        ),
        throw_errors=args.throw_errors,
    )
    pipeline = DatasetComplexityPipeline(
        kernel_size_list=args.kernel_size_list,
        save_dir=args.save_dir,
        classification_pipeline_kwargs=classification_pipeline_kwargs,
        n_samples=args.n_samples,
    )
    pipeline.run()
    pipeline.get_results_table(show=True)

    x_keys = [
        "kernel_size",
        # "n_features"
    ]
    for x_key in x_keys:
        y_keys = [
            ClassificationPipeline.FIT_TIME_KEY,
            # "kernel_n_ops",
            ClassificationPipeline.TEST_ACCURACY_KEY,
            # "kernel_depth",
        ] + [x for x in x_keys if x != x_key]
        n_rows = int(np.sqrt(len(y_keys)))
        n_cols = int(np.ceil(len(y_keys) / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.ravel(np.asarray([axes]))
        for i, y_key in enumerate(y_keys):
            pipeline.plot_results(x_axis_key=x_key, y_axis_key=y_key, fig=fig, ax=axes[i], show=False)
            axes[i].set_title(y_key)
        plt.tight_layout()
        if args.save_dir is not None:
            fig.savefig(os.path.join(args.save_dir, f"results_{x_key}.pdf"), bbox_inches="tight", dpi=900)
        plt.show()
        plt.close("all")


if __name__ == '__main__':
    sys.exit(main())
