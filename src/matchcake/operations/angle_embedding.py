import warnings
from collections import defaultdict
from functools import partial

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from ..utils import recursive_2in_operator
from .comp_rotations import CompRxRx, CompRyRy, CompRzRz

ROT = {"X": CompRxRx, "Y": CompRyRy, "Z": CompRzRz}
rotations_map = {  # TODO: to verify
    "XX": "I",
    "YY": "I",
    "ZZ": "I",
    "XY": "Z",
    "XZ": "Y",
    "YX": "Z",
    "YZ": "X",
    "ZX": "Y",
    "ZY": "X",
}
rotations_sign_map = defaultdict(lambda: 1j)
rotations_sign_map.update({"XY": 1j, "YX": -1j, "XZ": -1j, "ZX": 1j, "YZ": 1j, "ZY": -1j})


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
        rotations = hyperparameters.get("rotations", [ROT["X"]])
        contract_rots = hyperparameters.get("contract_rots", False)

        if contract_rots:
            warnings.warn("This method is not tested. Use at your own risk.", DeprecationWarning)
            op = partial(qml.math.einsum, "...ij,...jk->...ik")
            list_of_rots = [
                [
                    rot(
                        qml.math.stack([p0, p1], axis=-1),
                        wires=[wires[2 * i], wires[2 * i + 1]],
                    )
                    for rot in rotations
                ]
                for i, (p0, p1) in enumerate(zip(params[0::2], params[1::2]))
            ]
            return [recursive_2in_operator(op, rots) for rots in list_of_rots]

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
            "contract_rots": kwargs.get("contract_rots", False),
        }
        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    def simplify(self) -> "Operation":
        # TODO: to be tested
        # TODO: verify the correctness of the simplification
        # TODO: take a look at https://docs.pennylane.ai/en/stable/_modules/pennylane/transforms/optimization/merge_rotations.html
        rotations = self._rotations
        while len(rotations) > 1:
            new_rotations = []
            for i in range(0, len(rotations), 2):
                if i == len(rotations) - 1:
                    new_rotations.append(rotations[i])
                else:
                    new_rotations.append(rotations_map[rotations[i] + rotations[i + 1]])
            rotations = new_rotations
        if rotations[0] == "I":
            return qml.Identity(wires=self.wires)
        sign = rotations_sign_map[rotations[0]]
        # import torch
        # torch.allclose((self.decomposition()[0].gate_data @ self.decomposition()[1].gate_data, self.__class__(1j * self.data[0], wires=self.wires).decomposition()[0].gate_data))
        return self.__class__(sign * self.data[0], wires=self.wires)


class MAngleEmbeddings(Operation):
    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def _get_w0_idx_from_idx(idx, wires):
        return 2 * (idx % (len(wires) // 2))

    @staticmethod
    def _get_w0_from_idx(idx, wires):
        return wires[MAngleEmbeddings._get_w0_idx_from_idx(idx, wires)]

    @staticmethod
    def _get_w1_from_idx(idx, wires):
        return wires[MAngleEmbeddings._get_w0_idx_from_idx(idx, wires) + 1]

    @staticmethod
    def _get_layer_idx_from_idx(idx, wires):
        n_gates_per_layer = len(wires) // 2
        return idx // n_gates_per_layer

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        params = qml.math.concatenate(params, axis=-1)
        params = MAngleEmbeddings.pad_params(params)
        batched = qml.math.ndim(params) > 1
        params = qml.math.T(params) if batched else params
        wires = Wires(wires)
        rotations = hyperparameters.get("rotations", [ROT["X"]])
        return [
            rot(
                params=qml.math.stack([p0, p1], axis=-1),
                wires=[
                    MAngleEmbeddings._get_w0_from_idx(i, wires),
                    MAngleEmbeddings._get_w1_from_idx(i, wires),
                ],
                draw_label_params=f"p{i},l{MAngleEmbeddings._get_layer_idx_from_idx(i, wires)}",
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

    def __init__(self, features, wires, rotations="X", id=None):
        r"""
        Construct a new Matchgate AngleEmbedding operation.

        :Note: It is better to have one more wire than the number of features because every gate acts on two wires.

        :param features: The features to embed.
        :param wires: The wires to embed the features on.
        :param id: The id of the operation.
        """
        features = self.pad_params(features)
        self._rotations = rotations.split(",")
        self._hyperparameters = {"rotations": [ROT[r] for r in self._rotations]}
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)
