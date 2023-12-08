import os
import sys
import time
from typing import Optional, Union, List, Any, Tuple

import matplotlib
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import pythonbasictools as pbt

try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
from utils import MPL_RC_DEFAULT_PARAMS
from utils import get_device_memory_usage
from utils import init_nif_device, init_qubit_device
from msim import MatchgateOperation, mps


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
        MatchgateOperation(mg_params, wires=wires)
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
    max_wires_methods = {
        "nif.lookup_table": np.inf,
        "nif.explicit_sum": 10,
        "default.qubit": 20,
    }
    DEFAULT_N_WIRES = "linear"
    DEFAULT_N_GATES = "quadratic"
    DEFAULT_N_PROBS = "single"
    MISSING_VALUE = -1.0
    UNREACHABLE_VALUE = np.NaN
    DEFAULT_Y_LABELS = {
        "time": "Execution time [s]",
        "result": "Result [-]",
        "memory": "Memory [B]",
    }
    DEFAULT_NORMALIZED_Y_LABELS = {
        "time": "Normalized execution time [-]",
        "result": "Normalized result [-]",
        "memory": "Normalized memory size [-]",
    }
    DEFAULT_X_LABELS = {
        "n_wires": "Number of qubits [-]",
        "n_gates": "Number of gates [-]",
        "n_probs": "Number of probabilities [-]",
    }

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
            interface: str = "auto",
            use_cuda: bool = False,
    ):
        r"""


        :param n_variance_pts:
        :param n_pts:
        :param n_wires:
        :param n_gates:
        :param n_probs:
        :param methods:
        :param save_path:
        :param figures_folder:
        :param name:
        :param interface:
        :param use_cuda:
        """
        self.n_variance_pts = n_variance_pts
        self.n_pts = n_pts
        self.n_wires = n_wires
        self.n_gates = n_gates
        self.n_probs = n_probs
        self.methods = methods
        self.methods_functions = {
            "nif.lookup_table": self.execute_nif,
            "nif.explicit_sum": self.execute_nif,
            "default.qubit": self.execute_pennylane_qubit,
            "lightning.qubit": self.execute_pennylane_qubit,
        }
        self.device_init_functions = {
            "nif.lookup_table": self.init_nif_lookup_table,
            "nif.explicit_sum": self.init_nif_explicit_sum,
            "default.qubit": self.init_pennylane_default_qubit,
            "lightning.qubit": self.init_pennylane_lightning_qubit,
        }
        self.save_path = save_path
        self.figures_folder = figures_folder
        self.name = name
        self.interface = interface
        self.use_cuda = use_cuda

        self._init_n_pts_()
        self._init_n_wires_()
        self._init_n_gates_()
        self._init_n_probs_()
        self._init_methods_()

        self.wires_list = []
        self.parameters_list = []
        # self._init_wires_list_()
        # self._init_parameters_list_()

        self.result_data = np.full((len(self.methods), self.n_variance_pts, self.n_pts), self.MISSING_VALUE)
        self.time_data = np.full_like(self.result_data, self.MISSING_VALUE)
        self.memory_data = np.full_like(self.result_data, self.MISSING_VALUE)

    @property
    def all_data_generated(self):
        return np.all(np.logical_not(np.isclose(self.time_data, self.MISSING_VALUE)))

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
            gate_wires = self.get_wires(n_wires, n_gates)
            self.wires_list.append(gate_wires)
        return self.wires_list

    def get_wires(self, n_wires, n_gates):
        gate_wires = np.zeros((n_gates, 2), dtype=int)
        gate_wires[:, 0] = np.arange(n_gates, dtype=int) % (n_wires - 1)
        gate_wires[:, 1] = gate_wires[:, 0] + 1
        return gate_wires
    
    def _init_wires_list_block_wall_(self):
        self.wires_list = []
        for n_wires, n_gates in zip(self.n_wires, self.n_gates):
            pass
        raise NotImplementedError

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
        max_wires = self.max_wires_methods[self.methods[method_idx]]

        if self.n_wires[pt_idx] > max_wires:
            self.time_data[method_idx, variance_idx, pt_idx] = self.UNREACHABLE_VALUE
            self.result_data[method_idx, variance_idx, pt_idx] = self.UNREACHABLE_VALUE
            self.memory_data[method_idx, variance_idx, pt_idx] = self.UNREACHABLE_VALUE
        else:
            method_function = self.methods_functions[self.methods[method_idx]]
            device_init_function = self.device_init_functions[self.methods[method_idx]]
            n_wires, n_gates, n_probs = self.n_wires[pt_idx], self.n_gates[pt_idx], self.n_probs[pt_idx]
            wires = self.get_wires(n_wires, n_gates)
            params = mps.MatchgatePolarParams.random_batch_numpy(n_gates,  seed=variance_idx * pt_idx + pt_idx)
            params = self.push_params_to_interface(params)
            device = device_init_function(wires=n_wires)

            start_time = time.time()
            out = method_function(device, params, wires, n_probs)
            self.time_data[method_idx, variance_idx, pt_idx] = time.time() - start_time
            self.memory_data[method_idx, variance_idx, pt_idx] = get_device_memory_usage(device)
            self.result_data[method_idx, variance_idx, pt_idx] = np.NaN
        return dict(
            time=self.time_data[method_idx, variance_idx, pt_idx],
            result=self.result_data[method_idx, variance_idx, pt_idx],
            memory=self.memory_data[method_idx, variance_idx, pt_idx],
        )
    
    def _filter_unreachable_points_(self):
        mth_indexes = {
            mth: [tuple(index) for index in np.argwhere(self.n_wires > self.max_wires_methods.get(mth, np.inf))]
            for mth in self.methods
        }
        for mth, indexes in mth_indexes.items():
            if len(indexes) == 0:
                continue
            mth_idx = self.methods.index(mth)
            self.time_data[mth_idx, :, indexes] = self.UNREACHABLE_VALUE
            self.result_data[mth_idx, :, indexes] = self.UNREACHABLE_VALUE
            self.memory_data[mth_idx, :, indexes] = self.UNREACHABLE_VALUE

    def gen_all_data_points_(self, **kwargs):
        self._filter_unreachable_points_()
        indexes = np.argwhere(np.isclose(self.time_data, self.MISSING_VALUE))
        indexes = [tuple(index) for index in indexes]
        if len(indexes) == 0:
            return
        out_dict_list = pbt.apply_func_multiprocess(
            self.gen_data_point_,
            iterable_of_args=indexes,
            nb_workers=kwargs.get("nb_workers", 0),
            desc="Generating data points",
            verbose=kwargs.get("verbose", True),
            unit="pt",
        )
        for indexes, out_dict in zip(indexes, out_dict_list):
            self.time_data[indexes] = out_dict["time"]
            self.result_data[indexes] = out_dict["result"]
            self.memory_data[indexes] = out_dict["memory"]

    def init_pennylane_default_qubit(self, wires):
        return init_qubit_device(wires=wires, name="default.qubit")

    def init_pennylane_lightning_qubit(self, wires):
        return init_qubit_device(wires=wires, name="lightning.qubit")

    def init_nif_lookup_table(self, wires):
        return init_nif_device(wires=wires, prob_strategy="lookup_table")

    def init_nif_explicit_sum(self, wires):
        return init_nif_device(wires=wires, prob_strategy="explicit_sum")

    def execute_pennylane_qubit(
            self,
            device, params, wires, n_probs,
    ) -> Any:
        prob_wires = np.arange(n_probs)
        initial_binary_state = np.zeros(device.num_wires, dtype=int)
        qnode = qml.QNode(specific_matchgate_circuit, device, interface=self.interface)
        qubit_state = qnode(
            list(zip(params, wires)),
            initial_binary_state,
            all_wires=device.wires,
            in_param_type=mps.MatchgatePolarParams,
            out_op="state",
        )
        probs = msim.utils.get_probabilities_from_state(qubit_state, wires=prob_wires)
        return probs

    def execute_nif(self, device, params, wires, n_probs) -> Any:
        prob_wires = np.arange(n_probs)
        initial_binary_state = np.zeros(device.num_wires, dtype=int)
        qnode = qml.QNode(specific_matchgate_circuit, device, interface=self.interface)
        probs = qnode(
            list(zip(params, wires)),
            initial_binary_state,
            all_wires=device.wires,
            in_param_type=mps.MatchgatePolarParams,
            out_op="probs",
            out_wires=prob_wires,
        )
        return probs

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
        methods_colors = kwargs.get("methods_colors", plt.cm.get_cmap("tab10").colors)
        yaxis = kwargs.get("yaxis", "time")
        norm_y = kwargs.get("norm_y", False)
        default_yaxis_label = self.DEFAULT_Y_LABELS.get(yaxis, yaxis)
        if norm_y:
            default_yaxis_label = self.DEFAULT_NORMALIZED_Y_LABELS.get(yaxis, default_yaxis_label)
        yaxis_label = kwargs.get("yaxis_name", default_yaxis_label)
        xaxis = kwargs.get("xaxis", "n_wires")
        xaxis_label = kwargs.get("xaxis_name", self.DEFAULT_X_LABELS.get(xaxis, xaxis))
        std_coeff = kwargs.get("std_coeff", 1)
        methods = kwargs.get("methods", self.methods)
        methods_names = kwargs.get("methods_names", {k: k for k in methods})
        pt_indexes = kwargs.get("pt_indexes", np.arange(self.n_pts))
        for i, method in enumerate(methods):
            method_idx = self.methods.index(method)
            if yaxis == "time":
                y = self.time_data[method_idx][:, pt_indexes]
                max_y = np.nanmax(self.time_data)
            elif yaxis == "result":
                y = self.result_data[method_idx][:, pt_indexes]
                max_y = np.nanmax(self.result_data)
            elif yaxis == "memory":
                y = self.memory_data[method_idx][:, pt_indexes]
                max_y = np.nanmax(self.memory_data)
            else:
                raise ValueError(f"Unknown yaxis: {yaxis}.")
            y = np.where(np.isclose(y, self.UNREACHABLE_VALUE), np.NaN, y)
            if norm_y:
                y = y / max_y
            mean_y = np.nanmean(y, axis=0)
            std_y = std_coeff * np.nanstd(y, axis=0)
            if xaxis == "n_wires":
                x = np.asarray(self.n_wires, dtype=int)[pt_indexes]
            elif xaxis == "n_gates":
                x = np.asarray(self.n_gates, dtype=int)[pt_indexes]
            elif xaxis == "n_probs":
                x = np.asarray(self.n_probs, dtype=int)[pt_indexes]
            else:
                raise ValueError(f"Unknown xaxis: {xaxis}.")
            axes.plot(x, mean_y, label=methods_names.get(method, method), color=methods_colors[i])
            axes.fill_between(
                x, mean_y - std_y, mean_y + std_y, alpha=0.2, color=methods_colors[i]
            )
        n_ticks = kwargs.get("n_ticks", len(axes.get_xticks()))
        pt_indexes_ticks = np.linspace(0, len(pt_indexes) - 1, num=n_ticks, dtype=int, endpoint=True)
        pt_indexes_ticks = kwargs.get("pt_indexes_ticks", pt_indexes_ticks)
        if xaxis == "n_wires":
            axes.set_xticks(self.n_wires[pt_indexes][pt_indexes_ticks])
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
        axes.set_ylabel(yaxis_label)
        axes.set_title(kwargs.get("title", ""))
        std_patch = matplotlib.patches.Patch(color="gray", alpha=0.2, label=f"{std_coeff} Std")
        if kwargs.get("legend", True):
            lines, labels = axes.get_legend_handles_labels()
            axes.legend(handles=lines + [std_patch], labels=labels + [f"{std_coeff} Std"])
        if kwargs.get("tight_layout", False):
            fig.tight_layout()
        save_folder = kwargs.get("save_folder", None)
        if save_folder is not None:
            os.makedirs(f"{save_folder}", exist_ok=True)
            ext_list = ["pdf", "png"]
            filename = f"{yaxis}_vs_{xaxis}"
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
    def from_pickle(cls, pickle_path: str) -> "BenchmarkPipeline":
        import pickle
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
        
    @classmethod
    def from_pickle_or_new(cls, pickle_path: str, **kwargs):
        if os.path.exists(pickle_path):
            return cls.from_pickle(pickle_path)
        else:
            return cls(**kwargs)

    def push_params_to_interface(self, params):
        if self.interface == "auto":
            return params
        elif self.interface == "torch":
            import torch
            return torch.tensor(
                params,
                dtype=torch.float64,
                requires_grad=getattr(params, "requires_grad", False),
            ).to(device="cuda" if self.use_cuda else "cpu")
        elif self.interface == "jax":
            import jax
            return jax.numpy.asarray(params)
        elif self.interface == "tensorflow":
            import tensorflow as tf
            return tf.convert_to_tensor(params)
        elif self.interface == "numpy":
            return np.asarray(params)
        else:
            raise ValueError(f"Unknown interface: {self.interface}.")
