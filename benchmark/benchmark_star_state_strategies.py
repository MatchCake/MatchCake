import sys
from typing import Optional, Tuple, Dict

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import collections
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pythonbasictools as pbt

import matchcake as mc
from matchcake.devices.star_state_finding_strategies import (
    star_state_finding_strategy_map, get_star_state_finding_strategy
)
from matchcake.circuits import RandomSptmOperationsGenerator, RandomSptmHaarOperationsGenerator
from matchcake.utils.torch_utils import to_numpy


def get_state_distribution(
        circuit_gen: RandomSptmOperationsGenerator,
        **kwargs
):
    device = mc.NIFDevice(
        wires=circuit_gen.wires,
        show_progress=False,
        **kwargs
    )
    samples = device.execute_generator(
        circuit_gen,
        output_type="samples",
    )
    samples = np.reshape(to_numpy(samples), (-1, circuit_gen.n_qubits)).astype(int)
    samples_str = ["".join(str(x) for x in sample) for sample in samples]
    counts = collections.Counter(samples_str)
    sorted_keys = sorted(counts.keys())
    probs = np.array([counts[key] for key in sorted_keys]) / len(samples)
    sorted_probs_indexes = np.argsort(probs)[::-1]
    sorted_keys = [sorted_keys[i] for i in sorted_probs_indexes]
    sorted_probs = probs[sorted_probs_indexes]
    return sorted_probs, sorted_keys


def plot_state_distribution_from_samples(
        samples: np.ndarray,
        fig, ax,
        **kwargs
):
    samples = to_numpy(samples).astype(int)

    samples_reshaped = samples.reshape(-1, samples.shape[-2], samples.shape[-1])

    probs = []
    for bi in range(samples_reshaped.shape[0]):
        batch_samples = samples_reshaped[bi, :, :]

        unique_samples, unique_counts = np.unique(batch_samples, return_counts=True, axis=0)
        unique_probs = unique_counts / batch_samples.shape[0]
        probs.append(unique_probs)

    max_size = max(len(probs) for probs in probs) + 1
    probs = np.stack([np.concatenate([prob, np.zeros(max_size - len(prob))]) for prob in probs])
    keys = np.arange(max_size, dtype=int)

    # plot the distribution as a line plot and aggregate over seeds which is the first axis
    sns.lineplot(x=keys, y=probs.mean(axis=0), ax=ax)
    # plot ci as a shaded area
    ax.fill_between(
        keys,
        probs.mean(axis=0) - probs.std(axis=0),
        probs.mean(axis=0) + probs.std(axis=0),
        alpha=0.2,
    )

    # set y-axis to log scale
    ax.set_yscale("log")
    if kwargs.get("add_xlabel", True):
        ax.set_xlabel("State")
    else:
        ax.set_xlabel("")

    if kwargs.get("add_ylabel", True):
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("")

    if kwargs.get("add_title", True):
        ax.set_title("State Distribution")
    return fig, ax


def plot_state_distribution(
        probs: np.ndarray,
        keys: np.ndarray = None,
        **kwargs
):
    if keys is None:
        keys = np.arange(probs.shape[-1], dtype=int)

    keys_str = [str(key) for key in keys]
    seeds = [f"seed{seed}" for seed in range(probs.shape[0])]
    df = pd.DataFrame(probs, index=seeds, columns=keys_str)
    df["seed"] = seeds

    fig, ax = plt.subplots()
    # use sns to plot the distribution as a line plot
    if len(probs) == 1:
        sns.barplot(x=keys, y=probs, ax=ax)
    else:
        # plot the distribution as a line plot and aggregate over seeds which is the first axis
        sns.lineplot(x=keys, y=probs.mean(axis=0), ax=ax)
        # plot ci as a shaded area
        ax.fill_between(
            keys,
            probs.mean(axis=0) - probs.std(axis=0),
            probs.mean(axis=0) + probs.std(axis=0),
            alpha=0.2,
        )

    # set y-axis to log scale
    ax.set_yscale("log")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title("State Distribution")
    fig.savefig("figures/state_distribution.pdf")
    plt.show()


def make_state_distribution_plot(n_wires, n_ops, seeds, n_shots):
    outputs = pbt.multiprocessing_tools.apply_func_multiprocess(
        get_state_distribution,
        iterable_of_args=[
            (RandomSptmHaarOperationsGenerator(wires=n_wires, n_ops=n_ops, seed=seed),)
            for seed in seeds
        ],
        iterable_of_kwargs=[dict(shots=n_shots) for _ in seeds],
        desc="Generating state distributions",
        nb_workers=0,
    )
    probs_list, keys_list = zip(*outputs)

    max_size = max(len(probs) for probs in probs_list)
    probs_list = [np.concatenate([probs, np.zeros(max_size - len(probs))]) for probs in probs_list]
    probs = np.stack(probs_list, axis=0)
    plot_state_distribution(probs)


def get_star_state_and_time_from_strategy(
        device: mc.NIFDevice,
        strategy: str,
        **kwargs
):
    device.star_state_finding_strategy = get_star_state_finding_strategy(strategy)
    device._samples = None
    device._star_state = None
    device._star_probability = None
    start_time = time.perf_counter()
    star_state, star_prob = device.compute_star_state()
    if isinstance(star_state, np.ndarray):
        star_state = "".join(str(x) for x in star_state.astype(int))
    end_time = time.perf_counter()
    return star_state, star_prob, end_time - start_time, device.samples


def plot_star_state_finding_strategies_benchmark_for_n_wires(
        *,
        n_wires: int,
        seeds: np.ndarray,
        baseline_n_shots: int,
        n_shots: int,
        n_ops: int,
        batch_size: Optional[int],
        strategy_plot_name_map,
        columns_plot_name_map,
        all_strategies,
        fig, axes,
        **kwargs
):
    data = []
    p_bar = tqdm(total=len(seeds) * len(all_strategies), desc=f"Computing star states for {n_wires} wires")
    for seed in seeds:
        circuit_gen = RandomSptmHaarOperationsGenerator(wires=n_wires, n_ops=n_ops, seed=seed, batch_size=batch_size)
        device = mc.NIFDevice(
            wires=circuit_gen.wires,
            show_progress=False,
            shots=n_shots,
            contraction_strategy=None,
        )
        device.execute_generator(circuit_gen)

        for strategy in all_strategies:
            datum = dict()
            if strategy == "baseline":
                device.shots = baseline_n_shots
                star_state, star_prob, elapsed_time, samples = get_star_state_and_time_from_strategy(
                    device, "FromSampling"
                )
                device.shots = n_shots
                datum["samples"] = samples
            else:
                star_state, star_prob, elapsed_time, _ = get_star_state_and_time_from_strategy(device, strategy)
            datum.update(
                {
                    "strategy": strategy,
                    "star_state": star_state,
                    "star_prob": star_prob,
                    "elapsed_time": elapsed_time,
                    "seed": seed,
                }
            )
            data.append(datum)
            p_bar.set_postfix_str(f"seed={seed}, strategy={strategy}, elapsed_time={elapsed_time:.2f}")
            p_bar.update()

    p_bar.close()
    df = pd.DataFrame(data)
    # rename the strategies to the plot names
    df["strategy"] = df["strategy"].apply(lambda s: strategy_plot_name_map.get(s, s))
    baseline_column_name = strategy_plot_name_map.get("baseline", "baseline")
    all_strategies = [strategy_plot_name_map.get(s, s) for s in all_strategies]

    # rename the columns to the plot names
    df = df.rename(columns=columns_plot_name_map)
    strategy_column_name = columns_plot_name_map.get("strategy", "strategy")
    is_star_state_column_name = columns_plot_name_map.get("is_star_state", "is_star_state")
    elapsed_time_column_name = columns_plot_name_map.get("elapsed_time", "elapsed_time")
    samples_column_name = columns_plot_name_map.get("samples", "samples")
    star_state_column_name = columns_plot_name_map.get("star_state", "star_state")
    seed_column_name = columns_plot_name_map.get("seed", "seed")
    star_prob_column_name = columns_plot_name_map.get("star_prob", "star_prob")
    prob_error_column_name = columns_plot_name_map.get("prob_error", "prob_error")
    prob_accuracy_column_name = columns_plot_name_map.get("prob_accuracy", "prob_accuracy")

    # add the is_star_state column where the star_state is the same as the baseline star state at the same seed
    baseline_df = df[df[strategy_column_name] == baseline_column_name]
    df[is_star_state_column_name] = df.apply(
        lambda row: row[star_state_column_name] == baseline_df[
            baseline_df[seed_column_name] == row[seed_column_name]
            ][star_state_column_name].values[0],
        axis=1
    )

    # add the prob_error column which is the absolute difference between the star_prob and the baseline star_prob
    df[prob_error_column_name] = df.apply(
        lambda row: abs(row[star_prob_column_name] - baseline_df[
            baseline_df[seed_column_name] == row[seed_column_name]
            ][star_prob_column_name].values[0]),
        axis=1
    )
    # prob accuracy is 1 - |prob_error|
    df[prob_accuracy_column_name] = 1 - df[prob_error_column_name].abs()

    # use sns to show on lef the accuracy of the star state finding strategies
    all_strategies_wo_baseline = all_strategies[1:]
    sns.barplot(
        data=df[df[strategy_column_name] != baseline_column_name],
        x=strategy_column_name,
        y=prob_accuracy_column_name,
        ax=axes[0],
        capsize=0.1,
        order=all_strategies_wo_baseline
    )

    # use sns to show on right the elapsed time of the star state finding strategies
    sns.barplot(
        data=df,
        x=strategy_column_name,
        y=elapsed_time_column_name,
        ax=axes[1],
        capsize=0.1,
        order=all_strategies
    )

    if kwargs.get("add_title", True):
        axes[0].set_title(r"Probability Accuracy: $1 - |\Delta P|$")
        axes[1].set_title("Elapsed Time of Star State Finding Strategies")
    if kwargs.get("add_ylabel", True):
        axes[0].set_ylabel(r"$1 - |\Delta P|$ [-]")
        axes[1].set_ylabel("Elapsed Time [s]")
    else:
        axes[0].set_ylabel("")
        axes[1].set_ylabel("")

    if kwargs.get("add_xlabel", True):
        pass
    else:
        axes[0].set_xlabel("")
        axes[1].set_xlabel("")

    # put the probability distribution on the last plot
    plot_state_distribution_from_samples(
        to_numpy(df[df[strategy_column_name] == baseline_column_name][samples_column_name].values.tolist()),
        fig=fig, ax=axes[2],
        **kwargs
    )
    return fig, axes


def main():
    n_wires_list = [64, 128, 256]
    seeds = np.arange(10, dtype=int)
    baseline_n_shots = 256
    n_shots = 1
    n_ops = 10 * max(n_wires_list)
    batch_size = None

    strategy_plot_name_map = {
        "fromsampling": f"Sampling{n_shots}",
        "greedy": "Greedy",
        "baseline": f"Sampling{baseline_n_shots}",
    }
    columns_plot_name_map = {
        "star_state": "Star State",
        "star_prob": "Star Probability",
        "elapsed_time": "Elapsed Time [s]",
        "seed": "Seed",
        "is_star_state": "Is Star State",
        "samples": "Samples",
        "strategy": "Strategies",
        "prob_error": r"$\Delta P$",
        "prob_accuracy": "Accuracy",
    }

    # all_strategies = ["Greedy"]
    all_strategies = ["baseline"] + list(sorted(star_state_finding_strategy_map.keys()))

    fig, axes = plt.subplots(nrows=len(n_wires_list), ncols=3, figsize=(16, 6*len(n_wires_list)))
    axes = np.reshape(axes, (-1, 3))
    for i, n_wires in enumerate(n_wires_list):
        plot_star_state_finding_strategies_benchmark_for_n_wires(
            n_wires=n_wires,
            seeds=seeds,
            baseline_n_shots=baseline_n_shots,
            n_shots=n_shots,
            n_ops=n_ops,
            batch_size=batch_size,
            strategy_plot_name_map=strategy_plot_name_map,
            columns_plot_name_map=columns_plot_name_map,
            all_strategies=all_strategies,
            fig=fig, axes=axes[i, :],
            add_title=i == 0,
            add_xlabel=i == len(n_wires_list) - 1,
        )

    fig.savefig(f"figures/benchmark_star_state_finding_strategies_{baseline_n_shots}_{n_shots}.pdf")
    plt.show()
    return 0


if __name__ == '__main__':
    exit(main())
