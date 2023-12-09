import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires
from .m_rot import MRot
import numpy as np


class MAngleEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        batched = qml.math.ndim(params) > 1
        params = qml.math.T(params) if batched else params
        wires = Wires(wires)
        w0 = np.asarray(wires) % max(wires)
        wires_tuple = list(zip(w0, w0 + 1))
        return [MRot(params[i], wires=[w0, w1]) for i, (w0, w1) in enumerate(wires_tuple)]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data}, wires={self.wires.tolist()})"

    def __init__(self, features, wires, id=None):
        r"""
        Construct a new Matchgate AngleEmbedding operation.

        :Note: It is better to have one more wire than the number of features because every gate acts on two wires.

        :param features: The features to embed.
        :param wires: The wires to embed the features on.
        :param id: The id of the operation.
        """
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )
        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

