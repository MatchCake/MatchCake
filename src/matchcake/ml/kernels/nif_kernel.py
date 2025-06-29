import os
from typing import Optional

import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
from pennylane.ops.qubit.observables import BasisStateProjector

try:
    from pennylane.templates.broadcast import PATTERN_TO_NUM_PARAMS
except ImportError:
    # Hotfix for pennylane>0.39.0
    PATTERN_TO_NUM_PARAMS = {
        "pyramid": lambda w: (0 if len(w) in [0, 1] else sum(i + 1 for i in range(len(w) // 2))),
    }
from pennylane.wires import Wires

from matchcake.devices.nif_device import NonInteractingFermionicDevice
from matchcake.operations import MAngleEmbedding

from ...utils import torch_utils
from .kernel_utils import mrot_zz_template
from .ml_kernel import MLKernel


class NIFKernel(MLKernel):
    UNPICKLABLE_ATTRIBUTES = ["_device", "_qnode"]

    def __init__(self, size: Optional[int] = None, **kwargs):
        super().__init__(size=size, **kwargs)
        self._qnode = None
        self._device = None
        self.simpify_qnode = self.kwargs.get("simplify_qnode", False)
        self.qnode_kwargs = dict(
            interface=self.kwargs.get("interface", "torch" if self.use_cuda else "auto"),
            diff_method=self.kwargs.get("diff_method", None),
            cache=False,
        )
        self.qnode_kwargs.update(self.kwargs.get("qnode_kwargs", {}))
        self.device_workers = self.kwargs.get("device_workers", 0)
        self.device_kwargs = self.kwargs.get("device_kwargs", {})

    @property
    def wires(self):
        return Wires(list(range(self.size)))

    @property
    def n_ops(self):
        return self.get_n_ops()

    @property
    def n_params(self):
        return self.get_n_params()

    @property
    def qnode(self):
        if self._qnode is None and self.is_fitted:
            self.pre_initialize()
        return self._qnode

    @qnode.setter
    def qnode(self, qnode):
        self._qnode = qnode

    @property
    def tape(self):
        qnode = self.qnode
        if qnode is None:
            return None
        if getattr(qnode, "tape", None) is None and self.is_fitted:
            self.compile_qnode()
        return getattr(qnode, "tape", None)

    def cast_tensor_to_interface(self, tensor):
        if self.qnode.interface == "torch":
            tensor = torch_utils.to_tensor(tensor)
        else:
            tensor = torch_utils.to_numpy(tensor)
        if self.use_cuda:
            tensor = torch_utils.to_cuda(tensor)
        return tensor

    def initialize_parameters(self):
        super().initialize_parameters()
        if self._parameters is None:
            n_parameters = self.kwargs.get("n_parameters", PATTERN_TO_NUM_PARAMS["pyramid"](self.wires))
            self._parameters = [self.parameters_rng.uniform(0, 2 * np.pi, size=2) for _ in range(n_parameters)]
            self._parameters = np.array(self._parameters)
            if self.qnode.interface == "torch":
                import torch

                self._parameters = torch.from_numpy(self._parameters).float().requires_grad_(True)

    def pre_initialize(self):
        self.device_kwargs.setdefault("n_workers", getattr(self, "device_workers", 0))
        self._device = NonInteractingFermionicDevice(wires=self.size, **self.device_kwargs)
        self._qnode = qml.QNode(self.circuit, self._device, **self.qnode_kwargs)
        if self.simpify_qnode:
            self._qnode = qml.simplify(self.qnode)

    def compile_qnode(self):
        self.batch_distance(self.X_[:2], self.X_[:2])

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y)
        # TODO: optimize parameters with the given dataset
        # TODO: add kernel alignment optimization
        return self

    def circuit(self, x0, x1):
        MAngleEmbedding(x0, wires=self.wires)
        qml.broadcast(
            unitary=mrot_zz_template,
            pattern="pyramid",
            wires=self.wires,
            parameters=self.parameters,
        )
        qml.adjoint(MAngleEmbedding)(x1, wires=self.wires)
        qml.adjoint(qml.broadcast)(
            unitary=mrot_zz_template,
            pattern="pyramid",
            wires=self.wires,
            parameters=self.parameters,
        )
        projector: BasisStateProjector = qml.Projector(np.zeros(self.size), wires=self.wires)
        return qml.expval(projector)

    def single_distance(self, x0, x1, **kwargs):
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
        return self.qnode(x0, x1)

    def batch_distance(self, x0, x1, **kwargs):
        x0, x1 = self.cast_tensor_to_interface(x0), self.cast_tensor_to_interface(x1)
        return self.qnode(x0, x1)

    def get_n_ops(self):
        if self.tape is None:
            return None
        return len(self.tape.operations)

    def get_n_params(self):
        if self.tape is None:
            return None
        return len(self.tape.get_parameters())

    def draw(self, **kwargs):
        logging_func = kwargs.pop("logging_func", print)
        name = kwargs.pop("name", self.__class__.__name__)
        if getattr(self, "qnode", None) is None or getattr(self.qnode, "tape", None) is None:
            _str = f"{name}: "
        else:
            n_ops = len(self.qnode.tape.operations)
            n_params = len(self.qnode.tape.get_parameters())
            _str = f"{name} ({n_ops} ops, {n_params} params): "
        if self.is_fitted:
            _str += f"\n{qml.draw(self.qnode, **kwargs)(self.X_[0], self.X_[-1])}\n"
        else:
            _str += f"None"
        if logging_func is not None:
            logging_func(_str)
        return _str

    def draw_mpl(self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, **kwargs):
        x0, x1 = self.cast_tensor_to_interface(self.X_[:2]), self.cast_tensor_to_interface(self.X_[-2:])
        _fig, _ax = qml.draw_mpl(self.qnode, expansion_strategy=kwargs.get("expansion_strategy", "device"))(x0, x1)
        if fig is None or ax is None:
            fig, ax = _fig, _ax
        else:
            ax_position = ax.get_position()
            ax.remove()
            fig.axes.append(_ax)
            _ax.set_position(ax_position)
            _ax.figure = fig
            fig.add_axes(_ax)
            ax = _ax

        filepath: Optional[str] = kwargs.get("filepath", None)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath)

        if kwargs.get("show", False):
            plt.show()

        return fig, ax
