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
from numpy.polynomial import Polynomial
from scipy import stats
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, r2_score
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
from utils import MPL_RC_BIG_FONT_PARAMS, mStyles, find_complexity


class DatasetComplexityPipeline:
    DATASET_MIN_SIZE_MAP = {
        "digits": 2,
        "iris": 2,
        "breast_cancer": 2,
    }
    DATASET_MAX_SIZE_MAP = {
        "digits": 64,
        "iris": 4,
        "breast_cancer": 30,
    }
    DATASET_STEP_SIZE_MAP = {
        "digits": 4,
        "iris": 2,
        "breast_cancer": 2,
    }
    # DEFAULT_SIZE_LISTS = {
    #     d_name: np.arange(
    #         DATASET_MIN_SIZE_MAP[d_name], DATASET_MAX_SIZE_MAP[d_name], DATASET_STEP_SIZE_MAP[d_name]
    #     ).tolist()
    #     for d_name in DATASET_MAX_SIZE_MAP.keys()
    # }
    MTH_MAX_SIZE_MAP = {
        "fPQC-cuda": 16,
        "ifPQC-cuda": 16,
        "hfPQC-cuda": 16,
        "fPQC-cpu": np.inf,
        "ifPQC-cpu": np.inf,
        "hfPQC-cpu": np.inf,
        "PQC": 16,
    }

    @classmethod
    @property
    def default_size_lists(cls):
        return {
            d_name: np.arange(
                start=cls.DATASET_MIN_SIZE_MAP[d_name],
                stop=cls.DATASET_MAX_SIZE_MAP[d_name] + cls.DATASET_STEP_SIZE_MAP[d_name],
                step=cls.DATASET_STEP_SIZE_MAP[d_name],
            ).tolist()
            for d_name in cls.DATASET_MAX_SIZE_MAP.keys()
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
            kernel_size_list = self.default_size_lists.get(dataset_name, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
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
        should_run_pipelines = kwargs.pop("run_pipelines", True)
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
            p_bar.set_description(f"Complexity run on {self.dataset_name}, size={size}")
            cp_kwargs = deepcopy(self.classification_pipeline_kwargs)
            cp_kwargs["dataset_name"] = self.dataset_name
            cp_kwargs["kernel_kwargs"] = cp_kwargs.get("kernel_kwargs", {})
            cp_kwargs["kernel_kwargs"].update(dict(size=size))
            methods = self.filter_methods_by_max_size(cp_kwargs.get("methods", []), size)
            cp_kwargs["methods"] = methods
            cp_kwargs["dataset_n_samples"] = self.n_samples
            if save_dir is not None:
                cp_kwargs["save_path"] = os.path.join(
                    save_dir, f"{self.dataset_name}", f"size{size}", "cls.pkl"
                )
            self.classification_pipelines[size] = ClassificationPipeline.from_pickle_or_new(
                pickle_path=cp_kwargs.get("save_path", None), **cp_kwargs
            )
            if should_run_pipelines:
                self.classification_pipelines[size].run(**kwargs)
            self.classification_pipelines[size].save_all_results()
        p_bar.close()
        return self

    def get_results_table(self, **kwargs):
        df_list = []
        show = kwargs.pop("show", False)
        filepath: Optional[str] = kwargs.pop(
            "filepath", os.path.join(self.save_dir, self.dataset_name, "figures", f"results.csv")
        )
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

    def get_results_table_from_csvs(self, **kwargs):
        filepath: Optional[str] = kwargs.pop(
            "filepath", os.path.join(self.save_dir, self.dataset_name, "figures", f"results.csv")
        )
        root_folder = kwargs.get("folder", os.path.join(self.save_dir, self.dataset_name))
        csv_filename = kwargs.get("csv_filename", "mean_results_and_properties.csv")
        show = kwargs.pop("show", False)
        df_list = []
        for root, dirs, files in os.walk(root_folder):
            if csv_filename in files:
                csv_path = os.path.join(root, csv_filename)
                df_list.append(pd.read_csv(csv_path))
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
        confidence_interval = kwargs.get("confidence_interval", 0.10)
        df = self.results_table
        fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        kernels = kwargs.get("kernels", df[ClassificationPipeline.KERNEL_KEY].unique())
        y_scale_factor = kwargs.get("y_scale_factor", 1)
        kernel_to_color = kwargs.get("kernel_to_color", None)
        if kernel_to_color is None:
            kernel_to_color = {k: colors[i] for i, k in enumerate(kernels)}
        kernel_to_marker = kwargs.get("kernel_to_marker", None)
        if kernel_to_marker is None:
            kernel_to_marker = {k: mStyles[i] for i, k in enumerate(kernels)}
        kernel_to_linestyle = kwargs.get("kernel_to_linestyle", None)
        base_linestyle = kwargs.get("linestyle", "-")
        if kernel_to_linestyle is None:
            kernel_to_linestyle = {k: base_linestyle for k in kernels}
        kernel_to_lbl = kwargs.get("kernel_to_lbl", None)
        if kernel_to_lbl is None:
            kernel_to_lbl = {k: k for k in kernels}
        y_min, y_max = np.inf, -np.inf
        data_dict = {}
        kernel_to_polyfit = kwargs.get("polyfit", {})
        for i, (kernel_name, kernel_df) in enumerate(df.groupby(ClassificationPipeline.KERNEL_KEY)):
            if kernel_name not in kernels:
                continue
            x_sorted_unique = np.sort(kernel_df[x_axis_key].unique())
            y_series = kernel_df.groupby(x_axis_key)[y_axis_key]
            y_mean = y_scale_factor * y_series.mean().values
            y_std = y_scale_factor * y_series.std().values

            k_color = kernel_to_color.get(kernel_name, colors[i])
            k_marker = kernel_to_marker.get(kernel_name, mStyles[i])
            k_linestyle = kernel_to_linestyle.get(kernel_name, base_linestyle)

            if kernel_to_polyfit.get(kernel_name, False):
                *_, comp_out = self.add_complexity(
                    fig=fig, ax=ax,
                    x=x_sorted_unique,
                    all_x=np.sort(df[x_axis_key].unique()),
                    y=y_mean,
                    color=k_color,
                    alpha=0.5,
                    linestyle=":",
                )
                comp_lbl = comp_out["label"]
            else:
                comp_out, comp_lbl = None, ""
            pre_lbl, post_lbl = kwargs.get("pre_lbl", ""), kwargs.get("post_lbl", "")
            lbl = f"{pre_lbl}{kernel_to_lbl.get(kernel_name, kernel_name)}{post_lbl}{comp_lbl}"
            ax.plot(x_sorted_unique, y_mean, label=lbl, color=k_color, marker=k_marker, linestyle=k_linestyle)

            conf_int_a, conf_int_b = stats.norm.interval(
                confidence_interval, loc=y_mean,
                scale=y_std / np.sqrt(kernel_df.groupby(x_axis_key).size().values)
            )
            ax.fill_between(x_sorted_unique, conf_int_a, conf_int_b, alpha=0.2, color=k_color)
            y_min, y_max = min(y_min, np.nanmin(conf_int_a)), max(y_max, np.nanmax(conf_int_b))
            data_dict[kernel_name] = {
                "x": x_sorted_unique,
                "y": y_mean,
                "y_std": y_std,
                "y_min": y_min,
                "y_max": y_max,
                "conf_int_a": conf_int_a,
                "conf_int_b": conf_int_b,
                "complexity_out": comp_out,
                "complexity_lbl": comp_lbl,
            }

        ax.set_xlabel(kwargs.get("x_axis_label", x_axis_key))
        ax.set_ylabel(kwargs.get("y_axis_label", y_axis_key))
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        data_dict["y_min"], data_dict["y_max"] = y_min, y_max
        if kwargs.get("legend", True):
            ax.legend()
        if kwargs.get("show", False):
            plt.show()
        if kwargs.get("return_data_dict", False):
            return fig, ax, data_dict
        return fig, ax

    def plot_formatted_complexity_results(self, **kwargs):
        if kwargs.get("use_default_rc_params", True):
            plt.rcParams.update(MPL_RC_BIG_FONT_PARAMS)
        x_keys = {
            "N [-]": "kernel_size",
            # "n_features": "Number of features [-]",
        }
        accuracies_lbl = "Accuracies [%]"
        linestyles = ["-", "--", "-.", ":"]
        y_lbl_to_keys = {
            ClassificationPipeline.FIT_TIME_KEY: ClassificationPipeline.FIT_TIME_KEY,
            # "kernel_n_ops",
            accuracies_lbl: [ClassificationPipeline.TRAIN_ACCURACY_KEY, ClassificationPipeline.TEST_ACCURACY_KEY],
            # "kernel_depth",
        }
        y_lbl_to_pre_post_lbl = {
            accuracies_lbl: ("Train ", "Test "),
        }
        y_lbl_to_scale_factor = {
            ClassificationPipeline.FIT_TIME_KEY: 1,
            accuracies_lbl: 100,
        }
        df = self.results_table
        kernels = df[ClassificationPipeline.KERNEL_KEY].unique()
        gpu_methods = [m for m in kernels if "cuda" in m]
        cpu_methods = [m for m in kernels if m not in gpu_methods]
        poly_methods = [m for m in cpu_methods if m != "PQC"]
        exp_methods = [m for m in cpu_methods if m == "PQC"]
        set_methods = cpu_methods + [m for m in gpu_methods if m.replace("-cuda", "-cpu") not in cpu_methods]
        y_lbl_to_polyfit = {
            ClassificationPipeline.FIT_TIME_KEY: {**{m: True for m in cpu_methods}, **{m: False for m in gpu_methods}},
            accuracies_lbl: {m: False for m in kernels},
        }
        y_lbl_to_kernels_list = {
            ClassificationPipeline.FIT_TIME_KEY: kernels,
            accuracies_lbl: set_methods
        }
        gpu_linestyle, cpu_linestyle = "--", "-"
        train_linestyle, test_linestyle = "-", "-."
        kernel_to_linestyle = {
            m: gpu_linestyle if m in gpu_methods else cpu_linestyle
            for m in kernels
        }
        y_lbl_to_linestyle_list = {
            ClassificationPipeline.FIT_TIME_KEY: [kernel_to_linestyle, kernel_to_linestyle],
            accuracies_lbl: [{m: train_linestyle for m in kernels}, {m: test_linestyle for m in kernels}],
        }
        kernel_to_lbl = {m: m.replace("-cuda", "").replace("-cpu", "") for m in kernels}
        sorted_lbls = sorted(set(kernel_to_lbl.values()))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        kernel_lbl_to_color = {lbl: colors[i] for i, lbl in enumerate(sorted_lbls)}
        kernel_lbl_to_marker = {lbl: mStyles[i] for i, lbl in enumerate(sorted_lbls)}
        kernel_to_color = {k: kernel_lbl_to_color[lbl] for k, lbl in kernel_to_lbl.items()}
        kernel_to_marker = {k: kernel_lbl_to_marker[lbl] for k, lbl in kernel_to_lbl.items()}

        for x_lbl, x_key in x_keys.items():
            n_rows = int(np.sqrt(len(y_lbl_to_keys)))
            n_cols = int(np.ceil(len(y_lbl_to_keys) / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
            axes = np.ravel(np.asarray([axes]))
            data_dict = {}
            for i, (y_lbl, y_key) in enumerate(y_lbl_to_keys.items()):
                if not isinstance(y_key, list):
                    y_key = [y_key]
                y_min, y_max = np.inf, -np.inf
                for j, y_k in enumerate(y_key):
                    *_, j_data_dict = self.plot_results(
                        x_axis_key=x_key, x_axis_label="",
                        y_axis_key=y_k, y_axis_label=y_lbl,
                        kernels=y_lbl_to_kernels_list[y_lbl],
                        pre_lbl=y_lbl_to_pre_post_lbl.get(y_lbl, [""] * (j + 1))[j],
                        kernel_to_lbl=kernel_to_lbl,
                        kernel_to_color=kernel_to_color,
                        kernel_to_marker=kernel_to_marker,
                        y_scale_factor=y_lbl_to_scale_factor.get(y_lbl, 1),
                        linestyle=linestyles[j],
                        kernel_to_linestyle=y_lbl_to_linestyle_list[y_lbl][j],
                        polyfit=y_lbl_to_polyfit.get(y_lbl, {}),
                        fig=fig, ax=axes[i],
                        legend=False,
                        show=False,
                        return_data_dict=True,
                    )
                    data_dict[y_k] = j_data_dict
                    j_y_min, j_y_max = j_data_dict["y_min"], j_data_dict["y_max"]
                    y_min, y_max = min(y_min, j_y_min), max(y_max, j_y_max)

                x_min, x_max = np.nanmin(df[x_key].values), np.nanmax(df[x_key].values)
                axes[i].set_xlim(x_min, x_max)
                axes[i].set_ylim(y_min, y_max)

            # poly_y = np.mean([data_dict[ClassificationPipeline.FIT_TIME_KEY][m]["y"] for m in poly_methods], axis=0)
            # self.add_polyfit(
            #     fig=fig, ax=axes[0],
            #     x=data_dict[ClassificationPipeline.FIT_TIME_KEY][poly_methods[0]]["x"],
            #     y=poly_y,
            #     color="gray",
            #     linestyle="-.",
            # )
            fig.supxlabel(x_lbl)
            # create a legend with custom patches using the kernel_to_linestyle and kernel_to_color
            lbl_to_kernel = {v: k for k, v in kernel_to_lbl.items()}
            lbl_to_color_marker = {
                f"{lbl}{data_dict[ClassificationPipeline.FIT_TIME_KEY][lbl_to_kernel[lbl]]['complexity_lbl']}":
                    (c, kernel_to_marker[lbl_to_kernel[lbl]])
                for lbl, c in kernel_lbl_to_color.items()
            }
            patches = [
                plt.Line2D([0], [0], color=c, label=lbl, marker=mk)
                for lbl, (c, mk) in lbl_to_color_marker.items()
            ]
            patches += [
                plt.Line2D([0], [0], color="black", label=lbl, linestyle=ls)
                for lbl, ls in [("GPU", gpu_linestyle), ("CPU", cpu_linestyle), ("Polyfit", ":")]
            ]
            axes[0].legend(handles=patches, loc='upper left')
            patches = [
                plt.Line2D([0], [0], color="black", label=lbl, linestyle=ls)
                for lbl, ls in [("Train", train_linestyle), ("Test", test_linestyle)]
            ]
            axes[-1].legend(handles=patches, loc='lower right')
            fig.tight_layout()
            plt.tight_layout()
            if self.save_dir is not None:
                fig_save_path = os.path.join(self.save_dir, self.dataset_name, "figures", f"results_{x_key}.pdf")
                os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
                fig.savefig(fig_save_path, bbox_inches="tight", dpi=900)
            if kwargs.get("show", False):
                plt.show()
            plt.close("all")
        return

    def add_polyfit(self, **kwargs):
        fig: plt.Figure = kwargs.get("fig", None)
        ax: plt.Axes = kwargs.get("ax", None)
        if fig is None or ax is None:
            raise ValueError("fig and ax must be provided.")
        x = kwargs.get("x")
        y = kwargs.get("y")
        nan_mask = np.isnan(x) | np.isnan(y)
        x = x[~nan_mask]
        y = y[~nan_mask]

        all_x = kwargs.get("all_x", x)
        color = kwargs.get("color", "gray")
        alpha = kwargs.get("alpha", 0.5)
        linestyle = kwargs.get("linestyle", ":")

        poly_list = [Polynomial.fit(x, y, deg) for deg in x.astype(int)]
        r2_list = [r2_score(y, poly(x)) for poly in poly_list]
        complexity = x.astype(int)[np.argmax(r2_list)]

        poly = Polynomial.fit(x, y, complexity)
        complexity_coeff = poly.coef[0]
        x_fit = np.linspace(all_x[0], all_x[-1], 1_000)
        y_fit = poly(x_fit)
        r2 = r2_score(y, poly(x))
        output = {
            "poly": poly,
            "r2": r2,
            "poly_list": poly_list,
            "r2_list": r2_list,
            "complexity": complexity,
            "complexity_coeff": complexity_coeff,
        }

        ax.plot(x_fit, y_fit, color=color, linestyle=linestyle, alpha=alpha)
        return fig, ax, output

    def add_complexity(self, x, y, **kwargs):
        fig: plt.Figure = kwargs.get("fig", None)
        ax: plt.Axes = kwargs.get("ax", None)
        if fig is None or ax is None:
            raise ValueError("fig and ax must be provided.")
        nan_mask = np.isnan(x) | np.isnan(y)
        x = x[~nan_mask]
        y = y[~nan_mask]

        all_x = kwargs.get("all_x", x)
        color = kwargs.get("color", "gray")
        alpha = kwargs.get("alpha", 0.5)
        linestyle = kwargs.get("linestyle", ":")

        complexity_out = find_complexity(x, y, x_lbl="N")
        complexity_func = complexity_out["func"]
        complexity_lbl = complexity_out["label"]
        x_fit = np.linspace(all_x[0], all_x[-1], 1_000)
        y_fit = complexity_func(x_fit, *complexity_out["popt"])
        r2 = complexity_out["r_squared"]
        ax.plot(x_fit, y_fit, color=color, linestyle=linestyle, alpha=alpha)
        return fig, ax, complexity_out



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str,
        # default="digits",
        default="breast_cancer",
        help=f"The name of the dataset to be used for the classification. "
             f"Available datasets: {DatasetComplexityPipeline.DATASET_MAX_SIZE_MAP.keys()}."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=[
            "fPQC-cpu",
            "hfPQC-cpu",
            "ifPQC-cpu",
            "fPQC-cuda",
            "hfPQC-cuda",
            "ifPQC-cuda",
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
        default=os.path.join(os.path.dirname(__file__), "results_dc_cluster"),
        help="The directory where the results will be saved."
    )
    parser.add_argument(
        "--kernel_size_list", type=int, nargs="+",
        default=None,
        # default=[2, 4, 6, 8, 10, 12, 14,],
        help=f"The list of number of qubits to be used for the classification."
             f"Example: --kernel_size_list 2 4 8 16."
    )
    parser.add_argument(
        "--throw_errors", type=bool,
        default=False,
        # default=True,
        help="Whether to throw errors or not."
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--run_pipelines", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(["cuda" in m for m in args.methods]):
        matchcake.utils.cuda.is_cuda_available(throw_error=True, enable_warnings=True)
    print(f"{args.run_pipelines = }")
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
        dataset_name=args.dataset_name,
        save_dir=args.save_dir,
        classification_pipeline_kwargs=classification_pipeline_kwargs,
        n_samples=args.n_samples,
    )
    pipeline.run(run_pipelines=args.run_pipelines)
    plt.close("all")
    pipeline.get_results_table(show=True)
    # if args.run_pipelines:
    #     pipeline.run(run_pipelines=args.run_pipelines)
    #     plt.close("all")
    #     pipeline.get_results_table(show=True)
    # else:
    #     pipeline.get_results_table_from_csvs(show=True)
    plt.close("all")
    pipeline.plot_formatted_complexity_results(show=True)
    plt.close("all")


if __name__ == '__main__':
    # DatasetComplexityPipeline.DATASET_MAX_SIZE_MAP["breast_cancer"] = 18
    sys.exit(main())
