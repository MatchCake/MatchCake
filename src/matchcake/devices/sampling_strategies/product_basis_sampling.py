from typing import Callable, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import pennylane as qml
import torch
import tqdm
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ...utils.math import convert_and_cast_like
from ...utils.torch_utils import detach, to_tensor
from .sampling_strategy import SamplingStrategy


class ProductBasisSampling(SamplingStrategy):
    r"""Sampler for measurements in an arbitrary single-qubit product basis.

    Draws samples of a matchgate circuit applied to a product state, with each qubit :math:`k`
    measured along its own Bloch direction :math:`\hat{n}_k`, i.e. measuring the observable
    :math:`\hat{n}_k \cdot \vec{\sigma}_k`. Outcome bit ``0`` stands for the eigenvalue
    :math:`+1` and bit ``1`` for :math:`-1`, so ``(0, 0, 1)`` is the computational basis and
    ``(1, 0, 0)`` the X basis. Such measurements leave the matchgate set, since rotating a
    qubit to the X basis takes a Hadamard, but they can still be sampled efficiently
    [Brod2016]_ (Theorem 3 and Sec. III.1).

    The sampler follows Brod's construction. An ancilla in :math:`|+\rangle` is placed before
    the first qubit, with Majorana operators :math:`a_0 = X_A` and :math:`a_1 = Y_A`. On the
    states it meets the ancilla is always in :math:`|+\rangle`, so :math:`X_A = 1`. Its
    Majorana :math:`a_1` is the parity index of the extended covariance matrix
    :math:`\tilde\Lambda` (see
    :func:`~matchcake.devices.expval_strategies.m_pfaffian._extended_covariance.extended_covariance_matrix`),
    which therefore describes the product state with the ancilla as a Gaussian state. The
    circuit acts on it through the lifted single-particle transition matrix.

    The qubits are then measured one at a time, from the first to the last. When qubit
    :math:`k` is reached, the qubits :math:`l < k` are in computational basis states with
    known values of :math:`Z_l`, and on the current state

    .. math::

        X_k = -i\,\sigma_k\, a_1 c_{2k}, \qquad
        Y_k = -i\,\sigma_k\, a_1 c_{2k+1}, \qquad
        Z_k = -i\, c_{2k} c_{2k+1},

    where :math:`\sigma_k = \prod_{l<k} Z_l = \pm 1`. All three are quadratic in the Majorana
    operators :math:`(a_1, c_{2k}, c_{2k+1})`, and a single-qubit rotation of qubit
    :math:`k` acts as a rotation in :math:`SO(3)` of these three operators, which keeps the
    state Gaussian. Each step therefore:

    1. Rotates the triple :math:`(a_1, c_{2k}, c_{2k+1})` so that
       :math:`\hat{n}_k \cdot \vec{\sigma}_k` becomes :math:`-i\, c_{2k} c_{2k+1}`. The new
       :math:`a_1` is the direction :math:`w = (n_z, -\sigma_k n_y, \sigma_k n_x)` of the old
       triple.
    2. Measures :math:`Z_k` on the rotated state. The outcome :math:`\lambda = \pm 1` has
       probability :math:`(1 - \lambda \tilde\Lambda_{2k, 2k+1}) / 2`.
    3. Projects onto the outcome with the rank-2 update

       .. math::

           \tilde\Lambda_{ij} \leftarrow \tilde\Lambda_{ij}
           + \lambda \frac{\tilde\Lambda_{2k, i} \tilde\Lambda_{2k+1, j}
           - \tilde\Lambda_{2k, j} \tilde\Lambda_{2k+1, i}}{1 - \lambda \tilde\Lambda_{2k, 2k+1}},

       and updates :math:`\sigma_{k+1} = \lambda\, \sigma_k`.

    After the rotation qubit :math:`k` is left in a computational basis state, which is what
    makes :math:`\sigma_{k+1}` a number for the next step. The rotation and the update only
    change the probabilities of the qubits already measured, not those of the others, so the
    outcomes follow the distribution of the original measurement. With the rank-2 updates,
    a sample costs :math:`O(n^3)` and no Pfaffian is needed. When all the bases are the
    computational basis, the rotations are the identity and the sampler draws ordinary
    computational basis samples.

    The probability of a complete outcome is the product of the conditional probabilities
    and is computed exactly by :meth:`outcome_probabilities`. Marginal probabilities on a
    subset of the qubits are not provided, since whether they can be computed efficiently is
    open [Brod2016]_ (Sec. IV).

    :param shots_chunk_size: Maximum number of shots processed together. Every shot holds its
        own :math:`(2n+1) \times (2n+1)` matrix, so this bounds the memory used. Defaults to
        :attr:`DEFAULT_SHOTS_CHUNK_SIZE`.
    :type shots_chunk_size: Optional[int]

    .. [Brod2016] D. J. Brod, "Efficient classical simulation of matchgate circuits with
        generalized inputs and measurements," Phys. Rev. A 93, 062332 (2016),
        arXiv:1602.03539.
    """

    NAME = "ProductBasisSampling"
    DEFAULT_SHOTS_CHUNK_SIZE: int = 1024
    PAULI_BLOCH_VECTORS: Dict[str, Tuple[float, float, float]] = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }

    @staticmethod
    def measurement_frames(bloch_vectors: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        r"""Build the rotations that turn the measured observable into :math:`Z_k`.

        The rows of each rotation are the new :math:`(a_1, c_{2k}, c_{2k+1})` written in the
        old ones: :math:`w = (n_z, -\sigma n_y, \sigma n_x)`, then two unit vectors
        :math:`u, v` orthogonal to :math:`w` with :math:`u \times v = w`. The rotation is
        proper, and :math:`-i\,(u \cdot c)(v \cdot c) = \hat{n} \cdot \vec{\sigma}`.

        :param bloch_vectors: Unit Bloch vectors, shape (..., 3).
        :type bloch_vectors: torch.Tensor
        :param signs: The signs :math:`\sigma = \pm 1`, shape (...).
        :type signs: torch.Tensor
        :return: The rotations, shape (..., 3, 3).
        :rtype: torch.Tensor
        """
        bloch_vectors, signs = torch.broadcast_tensors(bloch_vectors, signs[..., None])
        signs = signs[..., 0]
        w = torch.stack(
            [bloch_vectors[..., 2], -signs * bloch_vectors[..., 1], signs * bloch_vectors[..., 0]], dim=-1
        )  # (..., 3)
        # The reference vector only needs to be far from parallel to w.
        reference = torch.zeros_like(w)
        use_second_axis = torch.abs(w[..., 1]) < 0.9
        reference[..., 1] = use_second_axis.to(w.dtype)
        reference[..., 2] = (~use_second_axis).to(w.dtype)
        u = reference - torch.sum(reference * w, dim=-1, keepdim=True) * w
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
        v = torch.linalg.cross(w, u, dim=-1)
        return torch.stack([w, u, v], dim=-2)

    @classmethod
    def pauli_bloch_vectors(cls, bases: Sequence[str]) -> np.ndarray:
        """Return the Bloch vectors of the given Pauli bases.

        :param bases: One Pauli letter (``"X"``, ``"Y"`` or ``"Z"``) per qubit.
        :type bases: Sequence[str]
        :return: The Bloch vectors, shape (n, 3).
        :rtype: np.ndarray
        :raises ValueError: If a letter is not ``"X"``, ``"Y"`` or ``"Z"``.
        """
        unknown = [basis for basis in bases if basis not in cls.PAULI_BLOCH_VECTORS]
        if unknown:
            raise ValueError(f"Unknown Pauli bases {unknown}. Expected 'X', 'Y' or 'Z'.")
        return np.array([cls.PAULI_BLOCH_VECTORS[basis] for basis in bases], dtype=float).reshape(-1, 3)

    def __init__(self, shots_chunk_size: Optional[int] = None):
        self.shots_chunk_size = shots_chunk_size or self.DEFAULT_SHOTS_CHUNK_SIZE

    def batch_generate_samples(
        self,
        device: qml.devices.QubitDevice,
        states_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]] = None,
        **kwargs,
    ) -> TensorLike:
        """Generate samples from the extended covariance matrix of the device.

        The probability callback is not used, since the probabilities come from the extended
        covariance matrix.

        :param device: Device providing ``num_wires``, ``shots`` and ``extended_covariance_matrix``.
        :type device: qml.devices.QubitDevice
        :param states_prob_func: Unused, kept for the :class:`SamplingStrategy` interface.
        :type states_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]]

        :keyword bloch_vectors: The Bloch vector of the measurement basis of each wire, in the
            order of the device wires, shape (num_wires, 3). Defaults to the computational basis.
        :keyword show_progress: Whether to display a progress bar. Defaults to False.

        :return: Samples of shape (shots, ..., num_wires).
        :rtype: TensorLike
        """
        bloch_vectors = kwargs.get("bloch_vectors", None)
        if bloch_vectors is None:
            bloch_vectors = self.pauli_bloch_vectors(["Z"] * device.num_wires)
        return self.sample(
            device.extended_covariance_matrix,
            bloch_vectors,
            device.shots,
            show_progress=kwargs.get("show_progress", False),
        )

    def generate_samples(
        self,
        device: qml.devices.QubitDevice,
        state_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]] = None,
        **kwargs,
    ) -> TensorLike:
        """Generate samples from the extended covariance matrix of the device.

        Same as :meth:`batch_generate_samples`, which takes the same keyword arguments.

        :param device: Device providing ``num_wires``, ``shots`` and ``extended_covariance_matrix``.
        :type device: qml.devices.QubitDevice
        :param state_prob_func: Unused, kept for the :class:`SamplingStrategy` interface.
        :type state_prob_func: Optional[Callable[[TensorLike, Wires], TensorLike]]
        :return: Samples of shape (shots, ..., num_wires).
        :rtype: TensorLike
        """
        return self.batch_generate_samples(device, state_prob_func, **kwargs)

    def outcome_probabilities(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
        outcomes: TensorLike,
    ) -> TensorLike:
        """Compute the exact probabilities of complete measurement outcomes.

        :param extended_covariance_matrix: The evolved extended covariance matrix,
            shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vector of the measurement basis of each qubit, shape (n, 3).
        :type bloch_vectors: TensorLike
        :param outcomes: Outcome bits of all the qubits, shape (m, n).
        :type outcomes: TensorLike
        :return: The probability of each outcome, shape (m, ...).
        :rtype: TensorLike
        """
        gamma, bloch_vectors_t = self._prepare_inputs(extended_covariance_matrix, bloch_vectors)
        outcomes_t = torch.as_tensor(np.asarray(qml.math.unwrap(outcomes)), dtype=torch.long)
        outcomes_t = outcomes_t.reshape(-1, outcomes_t.shape[-1])  # (m, n)
        if outcomes_t.shape[-1] != bloch_vectors_t.shape[0]:
            raise ValueError(
                f"Expected outcomes on {bloch_vectors_t.shape[0]} qubits. Got {outcomes_t.shape[-1]} instead."
            )
        batch_ndim = gamma.ndim - 2
        # (m, n) -> (m, ..., n), broadcast over the batch axes of the covariance matrix.
        outcomes_t = outcomes_t.reshape(outcomes_t.shape[:1] + (1,) * batch_ndim + outcomes_t.shape[1:])
        outcomes_t = outcomes_t.expand(outcomes_t.shape[:1] + gamma.shape[:-2] + outcomes_t.shape[-1:])
        _, probabilities = self._measure_sequentially(gamma, bloch_vectors_t, outcomes=outcomes_t)
        return convert_and_cast_like(probabilities, extended_covariance_matrix)

    def sample(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
        shots: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Draw samples in the given product basis.

        :param extended_covariance_matrix: The evolved extended covariance matrix,
            shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vector of the measurement basis of each qubit, shape (n, 3).
        :type bloch_vectors: TensorLike
        :param shots: The number of samples to draw.
        :type shots: int
        :param show_progress: Whether to display a progress bar over the shot chunks.
        :type show_progress: bool
        :return: Outcome bits, shape (shots, ..., n).
        :rtype: np.ndarray
        """
        gamma, bloch_vectors_t = self._prepare_inputs(extended_covariance_matrix, bloch_vectors)
        n_qubits = bloch_vectors_t.shape[0]
        batch_shape = tuple(gamma.shape[:-2])
        chunks = []
        for start in tqdm.trange(
            0,
            shots,
            self.shots_chunk_size,
            desc=f"[{self.NAME}] Generating Samples",
            disable=not show_progress,
            unit="chunk",
        ):
            n_chunk = min(self.shots_chunk_size, shots - start)
            uniforms = torch.from_numpy(np.random.random((n_chunk,) + batch_shape + (n_qubits,)))
            bits, _ = self._measure_sequentially(gamma, bloch_vectors_t, uniforms=uniforms)
            chunks.append(bits.numpy())
        if not chunks:
            return np.zeros((0,) + batch_shape + (n_qubits,), dtype=np.int64)
        return np.concatenate(chunks, axis=0)

    def _measure_sequentially(
        self,
        gamma: torch.Tensor,
        bloch_vectors: torch.Tensor,
        *,
        uniforms: Optional[torch.Tensor] = None,
        outcomes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Measure the qubits one after the other, drawing or imposing the outcomes.

        :param gamma: The extended covariance matrix, shape (..., 2n+1, 2n+1).
        :type gamma: torch.Tensor
        :param bloch_vectors: Unit Bloch vectors, shape (n, 3).
        :type bloch_vectors: torch.Tensor
        :param uniforms: Uniform random numbers used to draw the outcomes, shape (m, ..., n).
        :type uniforms: Optional[torch.Tensor]
        :param outcomes: Imposed outcome bits, shape (m, ..., n). Used when ``uniforms`` is None.
        :type outcomes: Optional[torch.Tensor]
        :return: The outcome bits, shape (m, ..., n), and their probabilities, shape (m, ...).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        draws = uniforms if uniforms is not None else outcomes
        assert draws is not None
        n_qubits = bloch_vectors.shape[0]
        # Move the ancilla to index 0, so that the qubit measured next always sits at indices 1
        # and 2. The rows and columns of a measured qubit are dropped, and the matrix shrinks by
        # two at each step.
        order = [2 * n_qubits] + list(range(2 * n_qubits))
        gamma = gamma[..., order, :][..., :, order]
        gamma = gamma.expand(draws.shape[:-1] + gamma.shape[-2:])  # (m, ..., 2n+1, 2n+1)
        signs = torch.ones(draws.shape[:-1], dtype=gamma.dtype)  # (m, ...)
        probabilities = torch.ones(draws.shape[:-1], dtype=gamma.dtype)  # (m, ...)
        bits = []
        for k in range(n_qubits):
            frames = self.measurement_frames(bloch_vectors[k], signs)  # (m, ..., 3, 3)
            # Rows of the new (ancilla, even, odd) triple, on the triple and on the rest.
            triple_rows = gamma[..., :3, :]  # (m, ..., 3, d)
            triple_block = frames @ triple_rows[..., :3] @ frames.transpose(-1, -2)  # (m, ..., 3, 3)
            rest_rows = frames @ triple_rows[..., 3:]  # (m, ..., 3, d - 3)

            correlation = triple_block[..., 1, 2]  # (m, ...)
            probability_of_zero = torch.clamp((1.0 - correlation) / 2.0, 0.0, 1.0)
            if uniforms is not None:
                bit = (uniforms[..., k] >= probability_of_zero).long()
            else:
                assert outcomes is not None
                bit = outcomes[..., k].long()
            eigenvalue = 1.0 - 2.0 * bit.to(gamma.dtype)  # (m, ...)
            probability = torch.where(bit == 0, probability_of_zero, 1.0 - probability_of_zero)
            probabilities = probabilities * probability
            bits.append(bit)

            # The matrix kept for the next steps, on the new ancilla and the unmeasured qubits.
            ancilla_row = rest_rows[..., 0, :]  # (m, ..., d - 3)
            corner = torch.zeros(ancilla_row.shape[:-1] + (1, 1), dtype=gamma.dtype)  # (m, ..., 1, 1)
            kept = torch.cat(
                [
                    torch.cat([corner, ancilla_row[..., None, :]], dim=-1),
                    torch.cat([-ancilla_row[..., :, None], gamma[..., 3:, 3:]], dim=-1),
                ],
                dim=-2,
            )  # (m, ..., d - 2, d - 2)
            row_even = torch.cat([triple_block[..., 1, :1], rest_rows[..., 1, :]], dim=-1)  # (m, ..., d - 2)
            row_odd = torch.cat([triple_block[..., 2, :1], rest_rows[..., 2, :]], dim=-1)  # (m, ..., d - 2)
            # Impossible imposed outcomes have a zero probability already. Their matrices are
            # left unchanged instead of dividing by zero.
            denominator = 2.0 * probability
            scale = torch.where(denominator > 0, eigenvalue / torch.where(denominator > 0, denominator, 1.0), 0.0)
            # Rank-2 update scale * (row_even row_odd^T - row_odd row_even^T) as a single batched product.
            left = scale[..., None, None] * torch.stack([row_even, row_odd], dim=-1)  # (m, ..., d - 2, 2)
            right = torch.stack([row_odd, -row_even], dim=-2)  # (m, ..., 2, d - 2)
            size = kept.shape[-1]
            gamma = torch.baddbmm(
                kept.reshape(-1, size, size), left.reshape(-1, size, 2), right.reshape(-1, 2, size)
            ).reshape(kept.shape)
            signs = signs * eigenvalue
        return torch.stack(bits, dim=-1), probabilities

    def _prepare_inputs(
        self,
        extended_covariance_matrix: TensorLike,
        bloch_vectors: TensorLike,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the inputs to float64 torch tensors and check their shapes.

        :param extended_covariance_matrix: The extended covariance matrix, shape (..., 2n+1, 2n+1).
        :type extended_covariance_matrix: TensorLike
        :param bloch_vectors: The Bloch vectors, shape (n, 3).
        :type bloch_vectors: TensorLike
        :return: The covariance matrix and the normalized Bloch vectors.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the shapes do not match or a Bloch vector is zero.
        """
        gamma = cast(torch.Tensor, to_tensor(detach(qml.math.real(extended_covariance_matrix)))).cpu()
        bloch_vectors_t = cast(torch.Tensor, to_tensor(detach(qml.math.real(bloch_vectors)))).cpu()
        if bloch_vectors_t.ndim != 2 or bloch_vectors_t.shape[-1] != 3:
            raise ValueError(f"The Bloch vectors must have shape (n, 3). Got {tuple(bloch_vectors_t.shape)} instead.")
        expected_dim = 2 * bloch_vectors_t.shape[0] + 1
        if gamma.ndim < 2 or tuple(gamma.shape[-2:]) != (expected_dim, expected_dim):
            raise ValueError(
                f"The extended covariance matrix must have shape (..., {expected_dim}, {expected_dim}) for "
                f"{bloch_vectors_t.shape[0]} qubits. Got {tuple(gamma.shape)} instead."
            )
        norms = torch.linalg.norm(bloch_vectors_t, dim=-1, keepdim=True)
        if bool(torch.any(norms < 1e-12)):
            raise ValueError("The Bloch vectors must be non-zero.")
        return gamma, bloch_vectors_t / norms
