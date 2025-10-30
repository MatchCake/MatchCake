import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import pennylane as qml
import torch
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from .sptm_comp_rxrx import SptmCompRxRx
from .sptm_comp_ryry import SptmCompRyRy
from .sptm_comp_rzrz import SptmCompRzRz

ROT = {"X": SptmCompRxRx, "Y": SptmCompRyRy, "Z": SptmCompRzRz}


class SptmAngleEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        params = qml.math.concatenate(params, axis=-1)
        params = SptmAngleEmbedding.pad_params(params)
        batched = qml.math.ndim(params) > 1
        params = qml.math.T(params) if batched else params
        wires = Wires(wires)
        rotations = hyperparameters.get("rotations", [ROT["X"]])

        return [
            rot(
                qml.math.stack([p0, p1], axis=-1),
                wires=[wires[2 * i], wires[2 * i + 1]],
            )
            for i, (p0, p1) in enumerate(zip(params[0::2], params[1::2]))
            for rot in rotations
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

    def __init__(self, features, wires, rotations="X", id=None, **kwargs):
        r"""
        Construct a new Matchgate AngleEmbedding operation.

        :Note: It is better to have one more wire than the number of features because every gate acts on two wires.

        :param features: The features to embed.
        :param wires: The wires to embed the features on.
        :param id: The id of the operation.

        :keyword contract_rots: If True, contract the rotations. Default is False.
        """
        features = self.pad_params(features)
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(f"Features must be of length {len(wires)} or less; got length {n_features}.")
        self._rotations = rotations.split(",")
        self._hyperparameters = {
            "rotations": [ROT[r] for r in self._rotations],
        }
        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)
