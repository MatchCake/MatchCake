import argparse
from typing import Optional

import pennylane as qml
from pennylane.wires import Wires

from ... import NonInteractingFermionicDevice
from .torch_model import TorchModel


class NIFTorchModel(TorchModel):
    DEFAULT_N_QUBITS = None
    MODEL_NAME = "NIFTorchModel"
    ATTRS_TO_HPARAMS = TorchModel.ATTRS_TO_HPARAMS + ["n_qubits"]

    @classmethod
    def add_model_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None):
        parent_parser = super().add_model_specific_args(parent_parser)
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group("NIF Torch Model")
        parser.add_argument(
            "--n_qubits",
            type=int,
            default=cls.DEFAULT_N_QUBITS,
            help="The number of qubits to use in the quantum kernel",
        )
        return parent_parser

    def __init__(self, *, n_qubits: Optional[int] = DEFAULT_N_QUBITS, **kwargs):
        kwargs.setdefault(
            "save_dir",
            self.default_save_dir_from_args({"n_qubits": n_qubits, **kwargs}),
        )
        super().__init__(**kwargs)
        self.n_qubits = n_qubits

        self.q_device = NonInteractingFermionicDevice(wires=self.n_qubits, show_progress=False)
        self.q_node = qml.QNode(
            self.circuit,
            self.q_device,
            interface="torch",
            diff_method="backprop",
            cache=False,
        )

    @property
    def wires(self):
        return Wires(list(range(self.n_qubits)))

    def circuit(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass.")
