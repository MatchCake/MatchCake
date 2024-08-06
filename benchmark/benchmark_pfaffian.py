import sys
from typing import Optional, Tuple, Dict

from matchcake.utils import pfaffian
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import os
from tqdm import tqdm
import pandas as pd


def compute_pfaffian(matrix, method, show_progress=False):
    start_time = time.perf_counter()
    pf = pfaffian(matrix, method=method, show_progress=show_progress)
    end_time = time.perf_counter()
    return end_time - start_time


def create_matrix(n: Optional[int], batch_size: Optional[int] = None, seed: int = 0, interface="numpy"):
    rn_gen = np.random.default_rng(seed)
    if batch_size is None:
        matrix = rn_gen.random((n, n))
    else:
        matrix = rn_gen.random((batch_size, n, n))
    matrix = matrix - np.einsum("...ij->...ji", matrix)

    if interface == "torch":
        matrix = torch.from_numpy(matrix)
    return matrix


def plot_results(results: pd.DataFrame, save_file: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # One color per method and one linestyle per interface
    method_to_color = {"det": "blue", "bLTL": "green", "bH": "red"}
    interface_to_linestyle = {"numpy": "--", "torch": "-"}

    for method in results["method"].unique():
        for interface in results["interface"].unique():
            mask = (results["method"] == method) & (results["interface"] == interface)
            # data needs to be sorted by n
            results[mask].sort_values("n", inplace=True)
            ax.plot(
                results[mask]["n"],
                results[mask]["time"],
                color=method_to_color[method],
                linestyle=interface_to_linestyle[interface],
                # label=f"{method} ({interface})"
            )
            # compute the std over the seeds
            time_std = results[mask].groupby("n")["time"].std()
            # ax.fill_between(
            #     results[mask]["n"].unique(), results[mask].groupby("n")["time"].mean() - time_std,
            #     results[mask].groupby("n")["time"].mean() + time_std, alpha=0.2
            # )
    patches = [
        plt.Line2D([0], [0], color="black", linestyle=ls, label=f"Interface: {interface}")
        for interface, ls in interface_to_linestyle.items()
    ]
    patches += [
        plt.Line2D([0], [0], color=color, linestyle="-", label=f"Method: {method}")
        for method, color in method_to_color.items()
    ]
    ax.legend(handles=patches)
    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)
    plt.show()


def main():
    save_file = os.path.join(os.path.dirname(__file__), "data", "benchmark_pfaffian.csv")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    batch_size_list = [1, ]
    n_list = 2 * np.linspace(1, 2048, num=1_000, dtype=int, endpoint=True)
    methods = ["det", "bLTL", "bH"]
    interfaces = [
        # "numpy",
        "torch"
    ]
    seeds = np.arange(0, 5, dtype=int)
    results = pd.DataFrame(columns=["n", "batch_size", "method", "interface", "seed", "time"])
    if os.path.exists(save_file):
        results = pd.read_csv(save_file)
    parameters_list = list(itertools.product(n_list, batch_size_list, methods, interfaces, seeds))
    save_freq = 100
    p_bar = tqdm(total=len(parameters_list), desc="Computing Pfaffian")
    for i, (n, batch_size, method, interface, seed) in enumerate(parameters_list):
        if results[
            (results["n"] == n) & (results["batch_size"] == batch_size) & (results["method"] == method) & (
                    results["interface"] == interface) & (results["seed"] == seed)
        ].shape[0] > 0:
            p_bar.update()
            continue
        matrix = create_matrix(n, batch_size, seed, interface)
        pf_time = compute_pfaffian(matrix, method)
        new_result = dict(n=n, batch_size=batch_size, method=method, interface=interface, seed=seed, time=pf_time)
        results = pd.concat([results, pd.DataFrame(new_result, index=[0])], ignore_index=True)
        if i % save_freq == 0:
            results.to_csv(save_file, index=False)
        p_bar.set_postfix_str(
            f"n={n}, batch_size={batch_size}, method={method}, interface={interface}, seed={seed}, time={pf_time}"
        )
        p_bar.update()
    p_bar.close()
    plot_results(results, save_file=os.path.join(os.path.dirname(__file__), "figures", "benchmark_pfaffian.pdf"))


if __name__ == "__main__":
    sys.exit(main())




