import sys
import argparse
import os
import sys
from copy import deepcopy
from fractions import Fraction
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy import stats
from tqdm import tqdm
import pickle
import difflib
from .utils import (
    MPL_RC_BIG_FONT_PARAMS,
    mStyles,
)

try:
    import matchcake
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    import matchcake

default_kernel_to_lbl = {
    "ifPQC": r"$\otimes$fPQC",
    "iPQC": r"$\otimes$PQC",
}


def gather_results(results_dir: str) -> pd.DataFrame:
    """
    Data folder structure:
    results_dir
    ├── size{s}
    │   ├── .class_pipeline
    │       ├── {kernel}
    │           ├── {f}
    │               ├── fit_time.pkl
    │               ├── test_metrics.pkl
    │               ├── train_metrics.pkl

    where {s} is the size of the kernel, {kernel} is the method used for the classification, and {f} is the fold index.

    The resulting dataframe will have the following columns:
    - kernel: the kernel used for the classification
    - kernel_size: the size of the kernel
    - fold: the fold index
    - train_accuracy: the accuracy of the training set
    - test_accuracy: the accuracy of the test set
    - fit_time: the time it took to fit the model

    :param results_dir: the directory where the results are stored
    :type results_dir: str
    :return: the results in a pandas dataframe
    :rtype: pd.DataFrame
    """
    results = []
    dot_class = ".class_pipeline"
    kernel_column = "kernel"
    kernel_size_column = "kernel_size"
    fold_column = "fold"
    fit_time_column = "fit_time"
    train_prefix = "train_"
    test_prefix = "test_"
    # Walk through the directory and when you find a .class_pipeline folder, read the results
    for size_folder in os.listdir(results_dir):
        size_folder_path = os.path.join(results_dir, size_folder, dot_class)
        if not os.path.isdir(size_folder_path):
            continue
        for kernel_folder in os.listdir(size_folder_path):
            kernel_folder_path = os.path.join(size_folder_path, kernel_folder)
            if not os.path.isdir(kernel_folder_path):
                continue
            for fold_folder in os.listdir(kernel_folder_path):
                fold_folder_path = os.path.join(kernel_folder_path, fold_folder)
                if not os.path.isdir(fold_folder_path):
                    continue
                train_metrics_file = os.path.join(fold_folder_path, "train_metrics.pkl")
                test_metrics_file = os.path.join(fold_folder_path, "test_metrics.pkl")
                fit_time_file = os.path.join(fold_folder_path, "fit_time.pkl")
                if not all([os.path.isfile(f) for f in [train_metrics_file, test_metrics_file, fit_time_file]]):
                    continue
                with open(train_metrics_file, "rb") as f:
                    train_metrics = pickle.load(f)
                with open(test_metrics_file, "rb") as f:
                    test_metrics = pickle.load(f)
                with open(fit_time_file, "rb") as f:
                    fit_time = pickle.load(f)
                data = {
                    kernel_column: kernel_folder,
                    kernel_size_column: size_folder,
                    fold_column: fold_folder,
                    fit_time_column: fit_time,
                }
                for key, value in train_metrics.items():
                    data[train_prefix + key] = value
                for key, value in test_metrics.items():
                    data[test_prefix + key] = value
                results.append(pd.DataFrame(data, index=[0]))
    df = pd.concat(results, ignore_index=True)
    df[kernel_size_column] = df[kernel_size_column].str.extract(r'size(\d+)').astype(int)
    return df


def merge_kernel_names(df: pd.DataFrame, kernel_map=None) -> pd.DataFrame:
    kernel_key = get_closest_match("kernel", df.columns)
    if kernel_map is None:
        kernel_map = {
            "fPQC": ["fPQC", "fPQC-cuda", "fPQC-cpu"],
            "ifPQC": ["ifPQC", "ifPQC-cuda", "ifPQC-cpu"],
            "hfPQC": ["hfPQC", "hfPQC-cuda", "hfPQC-cpu"],
        }
    for kernel_name, kernel_names in kernel_map.items():
        for k in kernel_names:
            df.loc[df[kernel_key] == k, kernel_key] = kernel_name
    return df


def remove_duplicate_folds(df: pd.DataFrame) -> pd.DataFrame:
    kernel_key = get_closest_match("kernel", df.columns)
    kernel_size_key = get_closest_match("kernel_size", df.columns)
    fold_key = get_closest_match("fold", df.columns)
    test_accuracy_key = get_closest_match("test_accuracy", df.columns)
    # we want to remove the duplicates but keep the one with the highest test accuracy
    df = df.sort_values([kernel_key, kernel_size_key, fold_key, test_accuracy_key], ascending=False)
    df = df.drop_duplicates([kernel_key, kernel_size_key, fold_key], keep="first")
    return df


def get_closest_match(word: str, possibilities: List[str]) -> str:
    return difflib.get_close_matches(word, possibilities, n=1)[0]


def plot_accuracies(df: pd.DataFrame, **kwargs):
    if kwargs.get("use_default_rc_params", True):
        plt.rcParams.update(MPL_RC_BIG_FONT_PARAMS)
        plt.rcParams["legend.fontsize"] = 22
        plt.rcParams["lines.linewidth"] = 4.0
        plt.rcParams["font.size"] = 36
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rc('text', usetex=True)

    df = merge_kernel_names(df)
    df = remove_duplicate_folds(df)

    x_key = kwargs.get("x_key", get_closest_match("kernel_size", df.columns))
    train_accuracy_key = kwargs.get("train_accuracy_key", get_closest_match("train_accuracy", df.columns))
    test_accuracy_key = kwargs.get("test_accuracy_key", get_closest_match("test_accuracy", df.columns))
    kernel_key = kwargs.get("kernel_key", get_closest_match("kernel", df.columns))
    x_lbl = kwargs.get("x_lbl", r"$N$ $[-]$")
    accuracies_lbl = r"Accuracies $[\%]$"
    y_keys = [train_accuracy_key, test_accuracy_key]
    linestyles = ["-", "-."]
    confidence_interval = kwargs.get("confidence_interval", 0.10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    kernels = list(sorted(kwargs.get("kernels", df[kernel_key].unique())))
    y_scale_factor = kwargs.get("y_scale_factor", 100)
    kernel_to_lbl = kwargs.get("kernel_to_lbl", default_kernel_to_lbl)
    kernel_to_color = {k: colors[i] for i, k in enumerate(kernels)}
    kernel_to_marker = {k: mStyles[i] for i, k in enumerate(kernels)}

    base_linestyle = kwargs.get("linestyle", "-")
    save_path = kwargs.get("save_path", None)

    fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))

    y_min, y_max = np.inf, -np.inf
    for j, y_axis_key in enumerate(y_keys):
        for i, (kernel_name, kernel_df) in enumerate(df.groupby(kernel_key)):
            x, y_series = list(zip(*[(x, group[y_axis_key]) for x, group in kernel_df.groupby(x_key)]))
            y_mean = y_scale_factor * np.array([y.mean() for y in y_series])
            y_std = y_scale_factor * np.array([y.std() for y in y_series])

            k_color = kernel_to_color.get(kernel_name, colors[i])
            k_marker = kernel_to_marker.get(kernel_name, mStyles[i])
            k_linestyle = linestyles[j] if j < len(linestyles) else base_linestyle
            kernel_lbl = kernel_to_lbl.get(kernel_name, kernel_name)

            conf_int_a, conf_int_b = stats.norm.interval(
                confidence_interval, loc=y_mean,
                scale=y_std / np.sqrt(kernel_df.groupby(x_key).size().values)
            )
            if kwargs.get("fill_between", True):
                ax.plot(x, y_mean, label=kernel_lbl, color=k_color, marker=k_marker, linestyle=k_linestyle)
                ax.fill_between(x, conf_int_a, conf_int_b, alpha=0.2, color=k_color)
            else:
                ax.errorbar(
                    x, y_mean, yerr=np.stack([y_mean - conf_int_a, conf_int_b - y_mean], axis=0),
                    marker=k_marker, color=k_color, linestyle=k_linestyle, fillstyle='full', label=kernel_lbl
                )

            y_min = min(y_min, np.nanmin(conf_int_a), np.nanmin(y_mean))
            y_max = max(y_max, np.nanmax(conf_int_b), np.nanmax(y_mean))

    ax.set_xlabel(x_lbl)
    ax.set_ylabel(accuracies_lbl)
    x_min, x_max = np.nanmin(df[x_key].values), np.nanmax(df[x_key].values)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min * 0.99, y_max * 1.01)
    ax.minorticks_on()
    if kwargs.get("legend", True):
        patches = [
            plt.Line2D(
                [0], [0],
                color=kernel_to_color.get(k, colors[i]),
                marker=kernel_to_marker.get(k, mStyles[i]),
                linestyle="-",
                label=kernel_to_lbl.get(k, k)
            )
            for i, k in enumerate(kernels)
        ]
        patches += [
            plt.Line2D([0], [0], color="black", label=lbl, linestyle=ls)
            for lbl, ls in zip(("Train", "Test"), linestyles)
        ]
        ax.legend(handles=patches, loc='lower right')
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=900)
    if kwargs.get("show", False):
        plt.show()
    return fig, ax

