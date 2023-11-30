import os

import numpy as np
from matplotlib import pyplot as plt

try:
    from .benchmark_pipeline import BenchmarkPipeline
except ImportError:
    from benchmark_pipeline import BenchmarkPipeline


def main():
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (n_gates, n_wires) in enumerate(zip(["2_power", "quadratic", "linear"], [10, 20, 20])):
        folder = f"data/results_{n_wires}qubits_{n_gates}_gates"
        benchmark_pipeline = BenchmarkPipeline(
            n_variance_pts=10,
            n_wires=np.linspace(2, n_wires, num=n_wires - 2, dtype=int),
            n_gates=n_gates,
            methods=[
                "nif.lookup_table",
                "default.qubit",
                "nif.explicit_sum"
            ],
            save_path=f"{folder}/objects",
        )
        benchmark_pipeline.run(
            nb_workers=-2,
            # overwrite=True,
        )
        benchmark_pipeline.show(
            fig=fig, ax=axes[0, i],
            xaxis="n_wires",
            std_coeff=3,
            # save_folder=os.path.join(folder, "figures"),
            show=False,
        )
        benchmark_pipeline.show(
            fig=fig, ax=axes[1, i],
            xaxis="n_gates",
            std_coeff=3,
            # save_folder=os.path.join(folder, "figures"),
            show=False,
        )
        axes[0, i].set_title(f"Gate distribution: {n_gates}")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        filepath = os.path.join("data", "figures", f"benchmark.{ext}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def main_nif():
    n_wires = 128
    n_gates = "quadratic"
    folder = f"data/results_{n_wires}qubits_{n_gates}_gates_nif_ceil"
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    benchmark_pipeline = BenchmarkPipeline.from_pickle_or_new(
        pickle_path=f"{folder}/objects.pkl",
        n_variance_pts=10,
        n_wires=np.linspace(2, n_wires, num=max(int(np.ceil(n_wires/10)), n_wires), dtype=int),
        n_gates=n_gates,
        methods=[
            "nif.lookup_table",
            "default.qubit",
            "nif.explicit_sum"
        ],
        save_path=f"{folder}/objects",
    )
    benchmark_pipeline.run(
        nb_workers=-2,
        # overwrite=True,
    )
    benchmark_pipeline.show(
        fig=fig, ax=axes[0],
        xaxis="n_wires",
        yaxis="time",
        std_coeff=3,
        # save_folder=os.path.join(folder, "figures"),
        show=False,
    )
    benchmark_pipeline.show(
        fig=fig, ax=axes[1],
        xaxis="n_wires",
        yaxis="memory",
        std_coeff=3,
        # save_folder=os.path.join(folder, "figures"),
        show=False,
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        filepath = os.path.join("data", "figures", f"benchmark.{ext}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # main()
    main_nif()
    # benchmark_pipeline = BenchmarkPipeline(
    #     n_variance_pts=10,
    #     n_wires=np.linspace(2, 128, num=2, dtype=int),
    #     n_gates="quadratic",
    #     methods=["nif.lookup_table"],
    # )
    # benchmark_pipeline.gen_data_point_(
    #     0, 0, -1
    # )
