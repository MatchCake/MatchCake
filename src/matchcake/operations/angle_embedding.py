import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from .comp_rotations import CompRxRx, CompRyRy, CompRzRz

ROT = {"X": CompRxRx, "Y": CompRyRy, "Z": CompRzRz}


class MAngleEmbedding(Operation):
    """
    Represents the Matchgate AngleEmbedding operation.

    This class defines a quantum operation to embed features into a quantum circuit
    using a matchgate-based angle embedding scheme. It is specifically designed to
    act on an even number of qubits and allows for customizable rotation operations
    to encode the input features. Features are padded if the number of them is odd.
    """

    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        """
        Computes the decomposition of quantum operations based on the provided parameters,
        wires, and hyperparameters. It concatenates and potentially pads input parameters,
        and applies specified rotations on target wires. Designed to handle both batched and
        unbatched input parameters.

        :param params: Positional arguments representing the parameters to be used in
            the decomposition. Each parameter can be a single value or a batch of values.
        :type params: list or numpy.ndarray
        :param wires: Specifies the wires on which the operations are applied.
        :type wires: list or pennylane.wires.Wires
        :param hyperparameters: Dictionary containing hyperparameters such as the types
            of rotations to apply. Defaults to a rotation around the X-axis if not
            provided in the dictionary key `"rotations"`.
        :type hyperparameters: dict
        :return: A list of operations representing the decomposition based on the applied
            rotations and input parameters.
        :rtype: list
        """
        params = qml.math.concatenate(params, axis=-1)
        params = MAngleEmbedding.pad_params(params)
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


class MAngleEmbeddings(Operation):
    r"""
    Represents a quantum operation for Matchgate Angle Embedding.

    This class embeds classical features into a quantum circuit using matchgate-based
    two-qubit rotations acting on *pairs* of wires. Features are padded with a trailing
    zero if an odd number is provided, since each matchgate consumes two parameters.

    Key difference vs :class:`MAngleEmbedding`
    -----------------------------------------
    ``MAngleEmbeddings`` supports *multi-layer* embeddings by **cycling over wire pairs**:
    if the number of feature-pairs exceeds the number of available wire-pairs
    (``len(wires)//2``), the operation **reuses the same wire pairs** and continues in a
    new logical "layer". Concretely, for wire pairs ``(wires[0], wires[1])``,
    ``(wires[2], wires[3])``, ... the i-th feature-pair is applied to pair
    ``i % (len(wires)//2)`` and its layer index is ``i // (len(wires)//2)``.

    By contrast, ``MAngleEmbedding`` performs a single pass over the wire pairs and
    raises an error when the (padded) feature length exceeds the number of provided
    wires (i.e., when an additional layer would be needed).

    Notes
    -----
    This operation requires at least two wires (so that at least one wire pair exists).

    The operation supports different rotation gates for embedding, which can be
    customized by specifying the rotations (e.g., "X", "X,Y,Z").
    """

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
