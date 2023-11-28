import os
import time
from typing import Optional, Union, List, Any

import matplotlib
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import pythonbasictools as pbt
from tqdm import tqdm

import msim
from utils import MPL_RC_DEFAULT_PARAMS
from msim import MatchgateOperator, mps
from tests.test_nif_device import init_nif_device, init_qubit_device

matplotlib.rcParams.update(MPL_RC_DEFAULT_PARAMS)


def specific_matchgate_circuit(params_wires_list, initial_state=None, **kwargs):
    all_wires = kwargs.get("all_wires", None)
    if all_wires is None:
        all_wires = set(sum([list(wires) for _, wires in params_wires_list], []))
    all_wires = np.sort(np.asarray(all_wires))
    if initial_state is None:
        initial_state = np.zeros(len(all_wires), dtype=int)
    qml.BasisState(initial_state, wires=all_wires)
    for params, wires in params_wires_list:
        mg_params = mps.MatchgatePolarParams.parse_from_any(params)
        MatchgateOperator(mg_params, wires=wires)
    out_op = kwargs.get("out_op", "state")
    if out_op == "state":
        return qml.state()
    elif out_op == "probs":
        return qml.probs(wires=kwargs.get("out_wires", None))
    else:
        raise ValueError(f"Unknown out_op: {out_op}.")


class BenchmarkPipeline:
    AVAILABLE_METHODS = {
        "nif.lookup_table",
        "nif.explicit_sum",
        "default.qubit",
        # "lightning.qubit",
    }
    DEFAULT_N_WIRES = "linear"
    DEFAULT_N_GATES = "quadratic"
    DEFAULT_N_PROBS = "single"

    def __init__(
            self,
            *,
            n_variance_pts: int = 10,
            n_pts: Optional[int] = None,
            n_wires: Optional[Union[int, List[int], str, np.ndarray]] = None,
            n_gates: Optional[Union[int, List[int], str, np.ndarray]] = None,
            n_probs: Optional[Union[int, List[int], str, np.ndarray]] = None,
            methods: Optional[Union[str, List[str]]] = None,
            save_path: Optional[str] = None,
            figures_folder: Optional[str] = None,
            name: Optional[str] = None,
    ):
        self.n_variance_pts = n_variance_pts
        self.n_pts = n_pts
        self.n_wires = n_wires
        self.n_gates = n_gates
        self.n_probs = n_probs
        self.methods = methods
        self.methods_functions = {
            "nif.lookup_table": self.execute_nif_lookup_table,
            "nif.explicit_sum": self.execute_nif_explicit_sum,
            "default.qubit": self.execute_default_qubit,
            "lightning.qubit": self.execute_lightning_qubit,
        }
        self.save_path = save_path
        self.figures_folder = figures_folder
        self.name = name

        self._init_n_pts_()
        self._init_n_wires_()
        self._init_n_gates_()
        self._init_n_probs_()
        self._init_methods_()

        self.wires_list = []
        self.parameters_list = []
        self._init_wires_list_()
        self._init_parameters_list_()

        self.result_data = np.full((len(self.methods), self.n_variance_pts, self.n_pts), np.NaN)
        self.time_data = np.full_like(self.result_data, np.NaN)

    @property
    def all_data_generated(self):
        return np.all(np.isfinite(self.time_data))

    def _init_n_pts_(self):
        length_list = []
        if isinstance(self.n_wires, (list, np.ndarray, tuple)):
            length_list.append(len(self.n_wires))
        if isinstance(self.n_gates, (list, np.ndarray, tuple)):
            length_list.append(len(self.n_gates))
        if isinstance(self.n_probs, (list, np.ndarray, tuple)):
            length_list.append(len(self.n_probs))
        if self.n_pts is not None:
            length_list.append(self.n_pts)
        if len(set(length_list)) == 0:
            self.n_pts = 10
        elif len(set(length_list)) == 1:
            self.n_pts = length_list[0]
        else:
            raise ValueError(
                "All lists must have the same length and n_pts must be equal to the length of the lists if it is given."
            )

    def _init_n_wires_(self):
        assert isinstance(self.n_pts, int), "n_pts must be initialized as an integer before n_wires."
        if self.n_wires is None:
            self.n_wires = self.DEFAULT_N_WIRES
        if isinstance(self.n_wires, str):
            self.n_wires = self._get_space(self.n_wires.lower(), constant=2, shift=2, dtype=int)
        if isinstance(self.n_wires, int):
            self.n_wires = self._get_constant_space(constant=self.n_wires, dtype=int)
        if isinstance(self.n_wires, (list, np.ndarray, tuple)):
            assert len(self.n_wires) == self.n_pts, "n_wires must have the same length as n_pts."
            self.n_wires = np.asarray(self.n_wires)

    def _init_n_gates_(self):
        assert isinstance(self.n_pts, int), "n_pts must be initialized as an integer before n_gates."
        assert isinstance(self.n_wires, np.ndarray), "n_wires must be initialized as a numpy array before n_gates."
        if self.n_gates is None:
            self.n_gates = self.DEFAULT_N_GATES
        if isinstance(self.n_gates, str):
            self.n_gates = self._get_space(self.n_gates.lower(), constant=self.n_wires, shift=self.n_wires, dtype=int)
        if isinstance(self.n_gates, int):
            self.n_gates = self._get_constant_space(constant=self.n_gates, dtype=int)
        if isinstance(self.n_gates, (list, np.ndarray, tuple)):
            assert len(self.n_gates) == self.n_pts, "n_gates must have the same length as n_pts."
            self.n_gates = np.asarray(self.n_gates)

    def _init_n_probs_(self):
        assert isinstance(self.n_pts, int), "n_pts must be initialized as an integer before n_probs."
        assert isinstance(self.n_wires, np.ndarray), "n_wires must be initialized as a numpy array before n_probs."
        if self.n_probs is None:
            self.n_probs = self.DEFAULT_N_PROBS
        if isinstance(self.n_probs, str):
            if self.n_probs.lower() == "single":
                self.n_probs = self._get_constant_space(constant=1, dtype=int)
            elif self.n_probs.lower() in ["all", "full"]:
                self.n_probs = 2 ** self.n_wires
            else:
                raise ValueError(
                    f"n_probs must be one of 'single', 'all', or 'full'. Got {self.n_probs}."
                )
        if isinstance(self.n_probs, int):
            self.n_probs = self._get_constant_space(constant=self.n_probs, dtype=int)
        if isinstance(self.n_probs, (list, np.ndarray, tuple)):
            assert len(self.n_probs) == self.n_pts, "n_probs must have the same length as n_pts."
            self.n_probs = np.asarray(self.n_probs)

    def _init_methods_(self):
        if self.methods is None:
            self.methods = self.AVAILABLE_METHODS
        if isinstance(self.methods, str):
            self.methods = [self.methods]
        if isinstance(self.methods, (np.ndarray, tuple)):
            self.methods = list(self.methods)
        assert all([method in self.AVAILABLE_METHODS for method in self.methods]), \
            f"methods must be one of {self.AVAILABLE_METHODS}. Got {self.methods}."

    def _init_wires_list_(self):
        r"""
        Generate the wires for the gates in the circuit as a brick wall circuit.

        :return: The wires for the gates in the circuit.
        """
        self.wires_list = []
        for n_wires, n_gates in zip(self.n_wires, self.n_gates):
            gate_wires = np.zeros((n_gates, 2), dtype=int)
            gate_wires[:, 0] = np.arange(n_gates, dtype=int) % (n_wires - 1)
            gate_wires[:, 1] = gate_wires[:, 0] + 1
            self.wires_list.append(gate_wires)
        return self.wires_list
    
    # def _init_wires_list_block_wall_(self):
    #     self.wires_list = []
    #     for n_wires, n_gates in zip(self.n_wires, self.n_gates):
    

    def _init_parameters_list_(self):
        r"""
        Generate the parameters for the gates in the circuit.

        :return: The parameters for the gates in the circuit.
        """
        self.parameters_list = pbt.apply_func_multiprocess(
            mps.MatchgatePolarParams.random_batch_numpy,
            iterable_of_args=[(self.n_variance_pts * n_gates, i) for i, n_gates in enumerate(self.n_gates)],
            nb_workers=-2,
            desc="Generating parameters",
            verbose=False,
            unit="p",
        )
        for i, (batch_params, n_gates) in enumerate(zip(self.parameters_list, self.n_gates)):
            params = batch_params.reshape((self.n_variance_pts, n_gates, mps.MatchgatePolarParams.N_PARAMS))
            self.parameters_list[i] = params
        return self.parameters_list

    def _get_linear_space(self, shift: int = 2, dtype: Any = int, **kwargs):
        return np.arange(self.n_pts, dtype=dtype) + shift

    def _get_constant_space(self, constant: int = 2, dtype: Any = int, **kwargs):
        return (constant * np.ones(self.n_pts, dtype=dtype)).astype(dtype)

    def _get_2_power_space(self, *args, **kwargs):
        return 2 ** self._get_linear_space(*args, **kwargs)

    def _get_quadratic_space(self, *args, **kwargs):
        return (self._get_linear_space(*args, **kwargs)) ** 2

    def _get_space(self, space_type: str, *args, **kwargs):
        if space_type.lower() == "constant":
            return self._get_constant_space(*args, **kwargs)
        elif space_type.lower() == "linear":
            return self._get_linear_space(*args, **kwargs)
        elif space_type.lower() == "2_power":
            return self._get_2_power_space(*args, **kwargs)
        elif space_type.lower() == "quadratic":
            return self._get_quadratic_space(*args, **kwargs)
        else:
            raise ValueError(
                f"space_type must be one of 'constant', 'linear', or '2_power'. Got {space_type}."
            )

    def gen_data_point_(self, method_idx: Union[int, str], variance_idx: int, pt_idx: int):
        if isinstance(method_idx, str):
            method_idx = self.methods.index(method_idx)
        method_function = self.methods_functions[self.methods[method_idx]]
        start_time = time.time()
        out = method_function(variance_idx, pt_idx)
        # self.result_data[method_idx, variance_idx, pt_idx] = out
        self.time_data[method_idx, variance_idx, pt_idx] = time.time() - start_time
        return self.time_data[method_idx, variance_idx, pt_idx]

    def gen_all_data_points_(self, **kwargs):
        iterable_of_args = list(np.ndindex(self.result_data.shape))
        times = pbt.apply_func_multiprocess(
            self.gen_data_point_,
            iterable_of_args=iterable_of_args,
            nb_workers=kwargs.get("nb_workers", 0),
            desc="Generating data points",
            verbose=kwargs.get("verbose", True),
            unit="pt",
        )
        for indexes, time_ in zip(iterable_of_args, times):
            self.time_data[indexes] = time_

    def execute_pennylane_qubit(self, variance_idx: int, pt_idx: int, device_name: str):
        params = self.parameters_list[pt_idx][variance_idx]
        wires = self.wires_list[pt_idx]
        n_wires = self.n_wires[pt_idx]
        n_probs = self.n_probs[pt_idx]
        prob_wires = np.arange(n_probs)
        initial_binary_state = np.zeros(n_wires, dtype=int)
        device = init_qubit_device(wires=n_wires, name=device_name)
        qnode = qml.QNode(specific_matchgate_circuit, device)
        qubit_state = qnode(
            list(zip(params, wires)),
            initial_binary_state,
            all_wires=device.wires,
            in_param_type=mps.MatchgatePolarParams,
            out_op="state",
        )
        probs = msim.utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
        return probs

    def execute_default_qubit(self, variance_idx: int, pt_idx: int):
        return self.execute_pennylane_qubit(variance_idx, pt_idx, device_name="default.qubit")

    def execute_lightning_qubit(self, variance_idx: int, pt_idx: int):
        return self.execute_pennylane_qubit(variance_idx, pt_idx, device_name="lightning.qubit")

    def execute_nif(self, variance_idx: int, pt_idx: int, prob_strategy: str):
        params = self.parameters_list[pt_idx][variance_idx]
        wires = self.wires_list[pt_idx]
        n_wires = self.n_wires[pt_idx]
        n_probs = self.n_probs[pt_idx]
        prob_wires = np.arange(n_probs)
        initial_binary_state = np.zeros(n_wires, dtype=int)
        device = init_nif_device(wires=n_wires, prob_strategy=prob_strategy)
        qnode = qml.QNode(specific_matchgate_circuit, device)
        probs = qnode(
            list(zip(params, wires)),
            initial_binary_state,
            all_wires=device.wires,
            in_param_type=mps.MatchgatePolarParams,
            out_op="probs",
            out_wires=prob_wires,
        )
        return probs

    def execute_nif_lookup_table(self, variance_idx: int, pt_idx: int):
        return self.execute_nif(variance_idx, pt_idx, prob_strategy="lookup_table")

    def execute_nif_explicit_sum(self, variance_idx: int, pt_idx: int):
        return self.execute_nif(variance_idx, pt_idx, prob_strategy="explicit_sum")

    def run(self, **kwargs):
        overwrite = kwargs.get("overwrite", False)
        if not overwrite:
            self.np_load()
        if not self.all_data_generated:
            self.gen_all_data_points_()
            self.np_save()
        self.to_pickle()

    def show(self, **kwargs):
        fig, axes = kwargs.get("fig", None), kwargs.get("axes", kwargs.get("ax", None))
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        if self.name is not None:
            fig.suptitle(self.name)
        methods_colors = ["tab:blue", "tab:orange", "tab:green"]
        xaxis = kwargs.get("xaxis", "n_wires")
        xaxis_label = kwargs.get("xaxis_name", f"Number of {xaxis} [-]")
        std_coeff = kwargs.get("std_coeff", 1)
        for i, method in enumerate(self.methods):
            mean_time = np.mean(self.time_data[i], axis=0)
            std_time = std_coeff * np.std(self.time_data[i], axis=0)
            if xaxis == "n_wires":
                x = np.asarray(self.n_wires, dtype=int)
            elif xaxis == "n_gates":
                x = np.asarray(self.n_gates, dtype=int)
            elif xaxis == "n_probs":
                x = np.asarray(self.n_probs, dtype=int)
            else:
                raise ValueError(f"Unknown xaxis: {xaxis}.")
            axes.plot(x, mean_time, label=method, color=methods_colors[i])
            axes.fill_between(
                x, mean_time - std_time, mean_time + std_time, alpha=0.2, color=methods_colors[i]
            )
        # axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        n_ticks = len(axes.get_xticks())
        pt_indexes = np.linspace(0, self.n_pts, num=n_ticks, dtype=int, endpoint=False)
        if xaxis == "n_wires":
            axes.set_xticks(self.n_wires[pt_indexes])
            axes.set_xticklabels([f"{int(n_wires)}" for n_wires in axes.get_xticks()])
        elif xaxis == "n_gates":
            # axes.set_xticks(self.n_gates[pt_indexes])
            # log2_n_gates = np.log2(np.asarray(axes.get_xticks()))
            # axes.set_xticklabels([r"$2^{"+f"{n_gates}"+r"}$" for n_gates in log2_n_gates])
            pass
        elif xaxis == "n_probs":
            # axes.set_xticks(self.n_probs)
            axes.set_xticklabels([f"{int(n_probs)}" for n_probs in axes.get_xticks()])
        else:
            raise ValueError(f"Unknown xaxis: {xaxis}.")
        axes.set_xlabel(xaxis_label)
        axes.set_ylabel("Execution time [s]")
        axes.set_title(kwargs.get("title", ""))
        std_patch = matplotlib.patches.Patch(color="gray", alpha=0.2, label=f"{std_coeff} Std")
        lines, labels = axes.get_legend_handles_labels()
        axes.legend(handles=lines + [std_patch], labels=labels + [f"{std_coeff} Std"])
        if kwargs.get("tight_layout", False):
            fig.tight_layout()
        save_folder = kwargs.get("save_folder", None)
        if save_folder is not None:
            os.makedirs(f"{save_folder}", exist_ok=True)
            ext_list = ["pdf", "png"]
            filename = f"time_vs_{xaxis}"
            for ext in ext_list:
                fig.savefig(f"{save_folder}/{filename}.{ext}", bbox_inches='tight', pad_inches=0.1, dpi=900)
        if kwargs.get("show", True):
            plt.show()
        return fig, axes

    def np_save(self):
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            np.savez(self.save_path, result_data=self.result_data, time_data=self.time_data)

    def np_load(self) -> "BenchmarkPipeline":
        if self.save_path is not None:
            save_path = self.save_path
            if not save_path.endswith(".npz"):
                save_path += ".npz"
            if not os.path.exists(save_path):
                return self
            data = np.load(save_path)
            assert np.allclose(self.result_data.shape, data["result_data"].shape), \
                f"result_data shape mismatch: {self.result_data.shape} != {data['result_data'].shape}"
            assert np.allclose(self.time_data.shape, data["time_data"].shape), \
                f"time_data shape mismatch: {self.time_data.shape} != {data['time_data'].shape}"
            self.result_data = data["result_data"]
            self.time_data = data["time_data"]
        return self

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
    def from_pickle(cls, pickle_path: str):
        import pickle
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
