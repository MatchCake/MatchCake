import os

import numpy as np
from matplotlib import pyplot as plt

try:
    from .benchmark_pipeline import BenchmarkPipeline
except ImportError:
    from benchmark_pipeline import BenchmarkPipeline


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (n_gates, n_wires) in enumerate(zip(["2_power", "quadratic", "linear"], [10, 20, 20])):
        folder = f"data/results_{n_wires}qubits_{n_gates}_gates"
        becnhmark_pipeline = BenchmarkPipeline(
            n_variance_pts=10,
            n_wires=np.linspace(2, n_wires, num=n_wires-2, dtype=int),
            n_gates=n_gates,
            methods=[
                "nif.lookup_table",
                "default.qubit",
            ],
            save_path=f"{folder}/objects",
        )
        becnhmark_pipeline.run(
            nb_workers=-2,
            # overwrite=True,
        )
        becnhmark_pipeline.show(
            fig=fig, ax=axes[0, i],
            xaxis="n_wires",
            # save_folder=os.path.join(folder, "figures"),
            show=False,
        )
        becnhmark_pipeline.show(
            fig=fig, ax=axes[1, i],
            xaxis="n_gates",
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
