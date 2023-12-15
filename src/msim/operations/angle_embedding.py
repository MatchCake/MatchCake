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
        params = qml.math.concatenate(params, axis=-1)
        params = MAngleEmbedding.pad_params(params)
        batched = qml.math.ndim(params) > 1
        params = qml.math.T(params) if batched else params
        wires = Wires(wires)
        return [
            MRot([p0, p1], wires=[wires[i], wires[i+1]])
            for i, (p0, p1) in enumerate(zip(params[0::2], params[1::2]))
        ]
    
    @staticmethod
    def pad_params(params):
        r"""
        If the number of parameters is odd, pad the parameters with zero to make it even.
        
        :param params: The parameters to pad.
        :return: The padded parameters.
        """
        n_params = qml.math.shape(params)[-1]
        if n_params % 2 != 0:
            params = qml.math.concatenate([params, qml.math.zeros_like(params[..., :1])], axis=-1)
        return params

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
        features = self.pad_params(features)
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

