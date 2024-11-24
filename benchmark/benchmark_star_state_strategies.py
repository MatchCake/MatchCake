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
from matchcake.circuits import RandomSptmOperationsGenerator
from matchcake.utils.torch_utils import to_numpy


class RandomSptmOperationsGeneratorV2(RandomSptmOperationsGenerator):
    def circuit(self):
        n_ops = 0
        rn_gen = np.random.default_rng(self.seed)
        while n_ops < self.n_ops:
            i = n_ops % (self.n_qubits - 1)
            yield mc.operations.SptmRzRz(
                mc.operations.SptmRzRz.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield mc.operations.SptmRyRy(
                mc.operations.SptmRyRy.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1
            yield mc.operations.SptmRzRz(
                mc.operations.SptmRzRz.random_params(self.batch_size),
                wires=[i, i + 1],
            )
            n_ops += 1

            if n_ops % self.n_qubits == 0:
                wire0 = rn_gen.choice(self.wires[:-1])
                wire1 = rn_gen.choice(self.wires[wire0+1:])
                yield mc.operations.SptmFSwap(wires=[wire0, wire1])
                n_ops += 1
        return

    def __iter__(self):
        return self.circuit()


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
            (RandomSptmOperationsGeneratorV2(wires=n_wires, n_ops=n_ops, seed=seed),)
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
    device._star_state = None
    device._star_probability = None
    start_time = time.perf_counter()
    star_state, star_prob = device.compute_star_state()
    if isinstance(star_state, np.ndarray):
        star_state = "".join(str(x) for x in star_state.astype(int))
    end_time = time.perf_counter()
    return star_state, end_time - start_time


def main():
    n_wires = 10
    seeds = np.arange(3, dtype=int)
    baseline_n_shots = 8192
    n_shots = 32
    n_ops = 10*n_wires
    batch_size = 3

    all_strategies = ["Greedy"]
    # all_strategies = ["baseline"] + list(star_state_finding_strategy_map.keys())

    # make_state_distribution_plot(n_wires, 10*n_wires, seeds, 8192)
    data = []
    p_bar = tqdm(total=len(seeds) * len(all_strategies), desc="Computing star states")
    for seed in seeds:
        circuit_gen = RandomSptmOperationsGeneratorV2(wires=n_wires, n_ops=n_ops, seed=seed, batch_size=batch_size)
        device = mc.NIFDevice(
            wires=circuit_gen.wires,
            show_progress=False,
            shots=n_shots,
        )
        device.execute_generator(circuit_gen)

        for strategy in all_strategies:
            if strategy == "baseline":
                device.shots = baseline_n_shots
                star_state, elapsed_time = get_star_state_and_time_from_strategy(device, "FromSampling")
                device.shots = n_shots
            else:
                star_state, elapsed_time = get_star_state_and_time_from_strategy(device, strategy)
            data.append(
                {
                    "strategy": strategy,
                    "star_state": star_state,
                    "elapsed_time": elapsed_time,
                    "seed": seed,
                }
            )
            p_bar.set_postfix_str(f"seed={seed}, strategy={strategy}, elapsed_time={elapsed_time:.2f}")
            p_bar.update()

    p_bar.close()

    df = pd.DataFrame(data)
    # add the is_star_state column where the star_state is the same as the baseline star state at the same seed
    baseline_df = df[df["strategy"] == "baseline"]
    df["is_star_state"] = df.apply(
        lambda row: row["star_state"] == baseline_df[baseline_df["seed"] == row["seed"]]["star_state"].values[0],
        axis=1
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # use sns to show on lef the accuracy of the star state finding strategies
    sns.barplot(data=df, x="strategy", y="is_star_state", ax=axes[0], capsize=0.1, order=all_strategies)

    # use sns to show on right the elapsed time of the star state finding strategies
    sns.barplot(data=df, x="strategy", y="elapsed_time", ax=axes[1], capsize=0.1, order=all_strategies)

    axes[0].set_title("Accuracy of Star State Finding Strategies")
    axes[0].set_ylabel("Accuracy [-]")
    axes[1].set_title("Elapsed Time of Star State Finding Strategies")
    axes[1].set_ylabel("Elapsed Time [s]")

    fig.savefig(f"figures/benchmark_star_state_finding_strategies_{n_wires}_wires.pdf")
    plt.show()
    return 0


if __name__ == '__main__':
    exit(main())
