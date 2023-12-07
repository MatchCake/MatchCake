import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pythonbasictools as pbt


try:
    from .benchmark_pipeline import BenchmarkPipeline
    from .utils import MPL_RC_DEFAULT_PARAMS
except ImportError:
    from benchmark_pipeline import BenchmarkPipeline
    from utils import MPL_RC_DEFAULT_PARAMS


def main(kwargs):
    matplotlib.rcParams.update(MPL_RC_DEFAULT_PARAMS)
    max_n_wires = [n for n in BenchmarkPipeline.max_wires_methods.values() if np.isfinite(n)]
    n_wires = list(sorted(set([2, 128, 1024, 2048, ] + max_n_wires)))
    # n_wires = np.linspace(2, 32, num=30, endpoint=True, dtype=int).tolist()
    n_wires = kwargs.get("n_wires", n_wires)
    n_wires_str = "-".join(map(str, n_wires))
    n_gates = kwargs.get("n_gates", 10 * max(n_wires))
    folder = kwargs.get("folder", f"data/results_{n_wires_str}-qubits_{n_gates}-gates_ceil")
    std_coeff = kwargs.get("std_coeff", 3)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    benchmark_pipeline = BenchmarkPipeline.from_pickle_or_new(
        pickle_path=f"{folder}/objects.pkl",
        n_variance_pts=kwargs.get("n_variance_pts", 3),
        n_wires=n_wires,
        n_gates=n_gates,
        methods=kwargs.get("methods", ["nif.lookup_table", "default.qubit", "nif.explicit_sum"]),
        save_path=f"{folder}/objects",
    )
    benchmark_pipeline.run(
        nb_workers=-2,
        overwrite=kwargs.get("overwrite", False),
    )
    benchmark_pipeline.show(
        fig=fig, ax=axes[0],
        xaxis="n_wires",
        yaxis="time",
        std_coeff=std_coeff,
        methods_names={"nif.lookup_table": "NIF", "default.qubit": "Pennylane", "nif.explicit_sum": "NIF.es"},
        # pt_indexes=[0, 1, 2],
        # n_ticks=kwargs.get("n_ticks", 3),
        pt_indexes_ticks=[0, -3, -2, -1],
        norm_y=True,
        legend=False,
        show=False,
    )
    benchmark_pipeline.show(
        fig=fig, ax=axes[1],
        xaxis="n_wires",
        yaxis="memory",
        std_coeff=std_coeff,
        methods_names={"nif.lookup_table": "NIF", "default.qubit": "Pennylane", "nif.explicit_sum": "NIF.es"},
        # pt_indexes=[0, 1, 2],
        # n_ticks=kwargs.get("n_ticks", 3),
        pt_indexes_ticks=[0, -3, -2, -1],
        norm_y=True,
        legend=True,
        show=False,
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        filepath = os.path.join(folder, "figures", f"benchmark.{ext}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    args = pbt.cmds.get_cmd_kwargs()
    main(args)
    # benchmark_pipeline = BenchmarkPipeline(
    #     n_variance_pts=1,
    #     n_wires=args.get("n_wires", [2, 128]),
    #     n_gates=args.get("n_gates", "linear"),
    #     methods=["nif.lookup_table"],
    # )
    # benchmark_pipeline.gen_all_data_points_(nb_workers=-2)
    # print(f"{benchmark_pipeline.memory_data[0, 0] = }")
    # benchmark_pipeline.show(
    #     xaxis="n_wires",
    #     yaxis="memory",
    #     std_coeff=3,
    #     show=True,
    # )
