import itertools
from functools import cached_property
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, cast

import numpy as np
import pennylane as qml
import torch
from pennylane.measurements import ExpectationMP, ProbabilityMP
from pennylane.operation import Operation, Operator, StatePrepBase
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.wires import Wires, WiresLike

from ..devices.expval_strategies.m_pfaffian import MPfaffianExpvalStrategy
from ..devices.expval_strategies.m_pfaffian._extended_covariance import extended_covariance_matrix
from ..devices.nif_device import NonInteractingFermionicDevice
from ..devices.probability_strategies import ProductStateProbabilityStrategy
from ..devices.sampling_strategies import get_sampling_strategy
from ..observables.batch_hamiltonian import BatchHamiltonian
from ..operations.single_particle_transition_matrices.single_particle_transition_matrix import (
    SingleParticleTransitionMatrixOperation,
)
from ..operations.state_preparation.product_state import ProductState
from ..typing import TensorLike
from ..utils._pfaffian import sector_pfaffian_features
from ..utils.covariance import block_diagonal_covariance, block_diagonalize_covariance
from ..utils.majorana import MajoranaGetter
from ..utils.math import complex_dtype_name_like, convert_and_cast_like, convert_like_and_cast_to, random_index
from ..utils.torch_utils import detach

ExecutionMode = Literal["parallel", "sequential", "direct"]
OutputType = Literal["expval", "probs", "samples"]


class MixedGaussianState:
    r"""
    A fermionic Gaussian state, possibly mixed, described by its Majorana covariance matrix.

    The state lives on ``m`` wires and is fully characterized by the real antisymmetric matrix
    :math:`\Lambda_{\mu\nu} = i\,\mathrm{Tr}[\rho\, c_\mu c_\nu]` (:math:`\mu \neq \nu`) built from the
    Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and :math:`c_{2k+1} = Z_{<k} Y_k` of the
    wires of the state. Such a state is what the partial trace of a matchgate circuit produces: see
    :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.partial_trace`.

    Block-diagonalizing the covariance matrix with :func:`~matchcake.utils.covariance.block_diagonalize_covariance`,

    .. math::
        \Lambda = R^\top \Big(\bigoplus_{j} (-\lambda_j) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\Big) R,

    exhibits the state as the product of single-mode mixtures
    :math:`\rho_j = \tfrac{1 + \lambda_j}{2} |0\rangle\langle 0| + \tfrac{1 - \lambda_j}{2} |1\rangle\langle 1|`
    evolved by the Gaussian unitary :math:`V_R` whose single-particle transition matrix is :math:`R`. Expanding
    the product over the computational basis gives the **pure-state decomposition**

    .. math::
        \rho = \sum_{x} w_x\, V_R |x\rangle\langle x| V_R^\dagger,
        \qquad
        w_x = \prod_j \frac{1 + (-1)^{x_j} \lambda_j}{2},

    where each member :math:`V_R|x\rangle` is a state the
    :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` can prepare with a
    :class:`~matchcake.operations.state_preparation.ProductState` followed by a
    :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`. Modes with :math:`|\lambda_j| = 1`
    are pure and contribute a single term, so the sum runs over :math:`2^{r}` members where :math:`r` is the
    number of mixed modes (at most the number of traced-out wires). The members and their weights are exposed
    by :meth:`pure_states`, :attr:`basis_states` and :attr:`weights`, and :meth:`execute` runs a follow-up
    circuit on every member, one at a time (``mode="sequential"``) or as one batched execution
    (``mode="parallel"``), and recombines the results with the weights.

    Because the covariance matrix is linear in the state, the same follow-up circuit can also be applied
    to the mixed state directly, :math:`\Lambda \mapsto R_2^\top \Lambda R_2` (:meth:`evolve`), and Pauli
    expectation values, probabilities and samples can be read from the evolved covariance matrix with the
    same Pfaffian formulas as for pure states (:meth:`expval`, :meth:`probs`, :meth:`sample`). This is what
    ``mode="direct"`` of :meth:`execute` does. It gives the same numbers as the ensemble modes, costs a single
    execution instead of :math:`2^r`, and its gradient is defined even when two mixed modes have exactly the
    same polarization, a point where the pure-state decomposition is not differentiable.

    The covariance matrix may carry a leading batch dimension, ``(B, 2m, 2m)``, for instance when the
    circuit that produced it had batched parameters. The batch is carried through every property and method.

    :param covariance_matrix: Real antisymmetric matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``.
    :type covariance_matrix: TensorLike
    :param wires: Labels of the ``m`` wires of the state. They must be consecutive integers because
        matchgates act on nearest-neighbour wires. Defaults to ``range(m)``.
    :type wires: Optional[WiresLike]
    :param atol: A mode whose polarization satisfies :math:`|\lambda_j| \geq 1 - \text{atol}` is treated
        as pure and contributes a single member to the decomposition. Defaults to :attr:`DEFAULT_ATOL`.
    :type atol: float
    :param device_kwargs: Keyword arguments (``r_dtype``, ``c_dtype``, ``pfaffian_chunk_size``, ...)
        forwarded to every :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` this
        state creates. Defaults to ``None`` (device defaults).
    :type device_kwargs: Optional[Dict[str, Any]]
    :param source_wires: Labels, in the original system, of the wires this state was reduced to. Purely
        informative; defaults to ``wires``.
    :type source_wires: Optional[WiresLike]
    """

    DEFAULT_ATOL = 1e-8
    ANTISYMMETRY_ATOL = 1e-6
    DEFAULT_SAMPLING_STRATEGY = NonInteractingFermionicDevice.DEFAULT_SAMPLING_STRATEGY

    @staticmethod
    def majorana_indices(wire_positions: Iterable[int]) -> np.ndarray:
        """Return the Majorana indices ``(2k, 2k+1)`` of every wire position, in order.

        :param wire_positions: Positions of the wires in the wire list of the state.
        :type wire_positions: Iterable[int]
        :return: Integer array of shape ``(2 * len(wire_positions),)``.
        :rtype: np.ndarray
        """
        positions = np.asarray(list(wire_positions), dtype=int)
        return np.stack([2 * positions, 2 * positions + 1], axis=-1).reshape(-1)

    @staticmethod
    def _entropy_term(probability: TensorLike) -> TensorLike:
        """Return ``p log p`` with the convention ``0 log 0 = 0``.

        :param probability: Probabilities in ``[0, 1]``.
        :type probability: TensorLike
        :return: The term, elementwise.
        :rtype: TensorLike
        """
        values: Any = probability
        positive = values > 0
        safe = qml.math.where(positive, values, qml.math.ones_like(values))
        return qml.math.where(positive, values * qml.math.log(safe), qml.math.zeros_like(values))

    @classmethod
    def from_polarizations(
        cls,
        polarizations: TensorLike,
        sptm: Optional[TensorLike] = None,
        wires: Optional[WiresLike] = None,
        **kwargs,
    ) -> "MixedGaussianState":
        r"""
        Build the state whose normal modes have the given polarizations.

        The covariance matrix is :math:`R^\top \Lambda_D R` with :math:`\Lambda_D` the block-diagonal
        matrix of :func:`~matchcake.utils.covariance.block_diagonal_covariance` and :math:`R` the given
        transition matrix (identity when omitted).

        :param polarizations: Polarizations :math:`\lambda_j \in [-1, 1]` of shape ``(m,)`` or ``(B, m)``.
        :type polarizations: TensorLike
        :param sptm: Single-particle transition matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``. Defaults to
            ``None`` (identity).
        :type sptm: Optional[TensorLike]
        :param wires: Labels of the wires. Defaults to ``range(m)``.
        :type wires: Optional[WiresLike]
        :param kwargs: Forwarded to the constructor.
        :return: The state.
        :rtype: MixedGaussianState
        """
        covariance = block_diagonal_covariance(polarizations)
        if sptm is not None:
            sptm = convert_and_cast_like(sptm, covariance)
            covariance = qml.math.einsum("...ij,...ik,...kl->...jl", sptm, covariance, sptm)
        return cls(covariance, wires=wires, **kwargs)

    def __init__(
        self,
        covariance_matrix: TensorLike,
        wires: Optional[WiresLike] = None,
        *,
        atol: float = DEFAULT_ATOL,
        device_kwargs: Optional[Dict[str, Any]] = None,
        source_wires: Optional[WiresLike] = None,
    ):
        shape = tuple(qml.math.shape(covariance_matrix))
        if len(shape) not in (2, 3) or shape[-1] != shape[-2] or shape[-1] % 2 != 0:
            raise ValueError(
                f"The covariance matrix must have shape (2m, 2m) or (B, 2m, 2m) with an even size, got {shape}."
            )
        transposed = qml.math.einsum("...ij->...ji", covariance_matrix)
        if not qml.math.allclose(covariance_matrix, -transposed, atol=self.ANTISYMMETRY_ATOL):
            raise ValueError("The covariance matrix must be real antisymmetric.")
        num_wires = shape[-1] // 2
        if wires is None:
            wires = range(num_wires)
        wires = Wires(wires)
        if len(wires) != num_wires:
            raise ValueError(f"Expected {num_wires} wires for a covariance matrix of shape {shape}, got {len(wires)}.")
        labels = wires.tolist()
        if any(not isinstance(label, (int, np.integer)) for label in labels) or labels != list(
            range(labels[0], labels[0] + num_wires)
        ):
            raise ValueError(
                f"The wires of a {type(self).__name__} must be consecutive integers, got {labels}. "
                f"Use the default wires or relabel them."
            )
        self._covariance_matrix = covariance_matrix
        self._wires = wires
        self._source_wires = wires if source_wires is None else Wires(source_wires)
        self.atol = float(atol)
        self.device_kwargs: Dict[str, Any] = dict(device_kwargs or {})

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_wires={self.num_wires}, batch_shape={self.batch_shape}, "
            f"num_mixed_modes={self.num_mixed_modes}, num_pure_states={self.num_pure_states})"
        )

    def density_matrix(self) -> TensorLike:
        r"""
        Dense :math:`2^m \times 2^m` density matrix of the state, from Wick's theorem.

        The operators :math:`c_S = c_{\mu_1} \cdots c_{\mu_{|S|}}` with :math:`\mu_1 < \cdots < \mu_{|S|}`
        form an orthogonal basis of the operator space, so

        .. math::
            \rho = \frac{1}{2^m} \sum_{|S| \text{ even}} i^{|S| / 2}\, \mathrm{Pf}(\Lambda|_S)\, c_S .

        The sum has :math:`2^{2m - 1}` terms: this method is meant for small systems, typically for
        verification.

        :return: Complex matrix of shape ``(2^m, 2^m)`` or ``(B, 2^m, 2^m)``.
        :rtype: TensorLike
        """
        covariance = self.covariance_matrix
        num_wires = self.num_wires
        majoranas = np.asarray(MajoranaGetter(num_wires).majorana_tensor, dtype=complex)  # (2m, 2^m, 2^m)
        dimension = 2**num_wires
        complex_dtype = complex_dtype_name_like(covariance)
        density = qml.math.cast(convert_like_and_cast_to(np.eye(dimension, dtype=complex), covariance), complex_dtype)
        density = density * qml.math.cast(
            convert_like_and_cast_to(np.ones(self.batch_shape + (1, 1)), covariance), complex_dtype
        )
        for size in range(2, 2 * num_wires + 1, 2):
            index_sets = np.array(list(itertools.combinations(range(2 * num_wires), size)), dtype=int)
            operators = np.stack(
                [np.linalg.multi_dot([majoranas[mu] for mu in subset]) for subset in index_sets], axis=0
            )  # (n_terms, 2^m, 2^m)
            pfaffians = sector_pfaffian_features(covariance, index_sets)  # (..., n_terms)
            pfaffians = qml.math.cast(convert_like_and_cast_to(pfaffians, covariance), complex_dtype)
            operators = qml.math.cast(convert_like_and_cast_to(operators, covariance), complex_dtype)
            density = density + (1j ** (size // 2)) * qml.math.einsum("...s,sij->...ij", pfaffians, operators)
        return density / dimension

    def entropy(self, base: Optional[float] = None) -> TensorLike:
        r"""
        Von Neumann entropy :math:`S(\rho) = -\mathrm{Tr}[\rho \ln \rho]` of the state.

        The normal modes are independent, so the entropy is the sum of the binary entropies of their
        occupation probabilities :math:`p_j = (1 + \lambda_j) / 2`. When the state is the partial trace of a
        pure state, this is the entanglement entropy between the kept and the traced-out wires.

        :param base: Base of the logarithm. Defaults to ``None`` (natural logarithm).
        :type base: Optional[float]
        :return: The entropy, of shape ``()`` or ``(B,)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        probabilities = 0.5 * (1.0 + polarizations)
        occupied_term: Any = self._entropy_term(probabilities)
        empty_term: Any = self._entropy_term(1.0 - probabilities)
        entropy = -qml.math.sum(occupied_term + empty_term, axis=-1)
        if base is not None:
            entropy = entropy / np.log(base)
        return entropy

    def ensemble_operations(self) -> List[Operation]:
        r"""
        Operations preparing every member of the pure-state decomposition as one batched state.

        The basis states of all the members (and of all the elements of the batch of the state, when it has
        one) are stacked into a single batched :class:`~matchcake.operations.state_preparation.ProductState`
        of batch size ``B * K``, followed by the transition-matrix operation of :math:`R` repeated to match.
        Prepending them to a circuit on a :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`
        runs that circuit on every member in one batched execution; the output of member ``k`` of batch
        element ``b`` sits at position ``b * K + k`` of the batch axis of the device.

        :return: A batched :class:`~matchcake.operations.state_preparation.ProductState` followed by a
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation`.
        :rtype: List[Operation]
        """
        basis = self.basis_states.reshape(-1, self.num_wires)  # (B * K, m)
        product_state = ProductState.from_basis_state(basis, wires=self.wires)
        sptm = self.sptm
        if len(self.batch_shape) > 0:
            num_members = self.num_pure_states
            sptm = qml.math.reshape(
                qml.math.stack([sptm] * num_members, axis=1), (basis.shape[0], 2 * self.num_wires, 2 * self.num_wires)
            )
        return [product_state, SingleParticleTransitionMatrixOperation(sptm, wires=self.wires)]

    def evolve(self, operations: Iterable[Operation], **device_kwargs) -> "MixedGaussianState":
        r"""
        Apply a matchgate circuit to the state, at the level of the covariance matrix.

        The circuit is compiled to its single-particle transition matrix :math:`R_2` by a
        :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` on the wires of the state, and
        the covariance matrix is mapped to :math:`R_2^\top \Lambda R_2`. Since the map is linear in the
        state, the result is exactly the mixture of the evolved members of :meth:`pure_states`.

        :param operations: Operations of the circuit, acting on the wires of the state. State preparations
            are not allowed.
        :type operations: Iterable[Operation]
        :param device_kwargs: Overrides of the device keyword arguments of the state.
        :return: The evolved state, on the same wires.
        :rtype: MixedGaussianState
        """
        operations = self._check_operations(operations)
        device = self._make_device(**device_kwargs)
        device.apply(operations)
        circuit_sptm = device.global_sptm.matrix(self.wires)
        covariance = self.covariance_matrix
        if isinstance(circuit_sptm, torch.Tensor):
            covariance = convert_and_cast_like(covariance, circuit_sptm)
        else:
            circuit_sptm = convert_and_cast_like(circuit_sptm, covariance)
        evolved = qml.math.einsum("...ij,...ik,...kl->...jl", circuit_sptm, covariance, circuit_sptm)
        return self._new(evolved, wires=self.wires, source_wires=self.source_wires)

    def execute(
        self,
        operations: Iterable[Operation],
        observable: Optional[Operator] = None,
        output_type: Optional[OutputType] = None,
        *,
        mode: ExecutionMode = "parallel",
        shots: Optional[int] = None,
        device_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Any]:
        r"""
        Run a follow-up circuit on the state and return the requested output.

        With ``mode="parallel"`` every member of the pure-state decomposition is prepared as one element of a
        batched :class:`~matchcake.operations.state_preparation.ProductState`, the circuit is executed once
        on a :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice` and the per-member outputs
        are recombined with :attr:`weights`. With ``mode="sequential"`` the members are executed one after the
        other. With ``mode="direct"`` the circuit is applied to the covariance matrix (:meth:`evolve`) and the
        output is read from it without enumerating the members.

        The parallel mode requires a circuit without a batch dimension of its own, because the batch axis
        of the device is used for the members (and for the batch of the state, when it has one). A circuit
        with batched parameters runs in the sequential or the direct mode.

        Analytic outputs (``shots=None``) are exact. With a finite number of ``shots``, ``"samples"`` draws
        the member of every shot from the weights and then a computational-basis sample from that member,
        and ``"expval"`` and ``"probs"`` are estimated from such samples exactly like the device does. As with
        the device, a sample-based expectation value is only meaningful for an observable that is diagonal
        in the computational basis.

        :param operations: Operations of the follow-up circuit, acting on the wires of the state.
        :type operations: Iterable[Operation]
        :param observable: Observable for ``output_type="expval"``.
        :type observable: Optional[Operator]
        :param output_type: ``"expval"``, ``"probs"`` or ``"samples"``. Defaults to ``None`` (nothing is
            computed and ``None`` is returned).
        :type output_type: Optional[OutputType]
        :param mode: ``"parallel"``, ``"sequential"`` or ``"direct"``. Defaults to ``"parallel"``.
        :type mode: ExecutionMode
        :param shots: Number of shots for sample-based outputs. Defaults to ``None`` (analytic).
        :type shots: Optional[int]
        :param device_kwargs: Overrides of the device keyword arguments of the state.
        :type device_kwargs: Optional[Dict[str, Any]]
        :param kwargs: Forwarded to :meth:`~matchcake.devices.nif_device.NonInteractingFermionicDevice.execute_output`,
            for instance ``wires`` for ``output_type="probs"``.
        :return: The recombined output. For ``"expval"`` and ``"probs"`` the leading dimensions are the batch
            of the state (if any) followed by the batch of the circuit (sequential and direct modes only);
            ``"samples"`` has shape ``(shots, *batch, m)``.
        :rtype: Optional[Any]
        """
        if output_type is None:
            return None
        if output_type not in ("expval", "probs", "samples"):
            raise ValueError(
                f"Output type {output_type!r} is not supported by {type(self).__name__}.execute. "
                f"Supported output types are 'expval', 'probs' and 'samples'."
            )
        if mode not in ("parallel", "sequential", "direct"):
            raise ValueError(f"Unknown execution mode {mode!r}. Use 'parallel', 'sequential' or 'direct'.")
        operations = self._check_operations(operations)
        merged_device_kwargs = {**self.device_kwargs, **(device_kwargs or {})}

        if shots is not None and output_type != "samples":
            samples = cast(
                np.ndarray,
                self.execute(
                    operations, output_type="samples", mode=mode, shots=shots, device_kwargs=merged_device_kwargs
                ),
            )
            return self._process_samples(samples, observable=observable, output_type=output_type, **kwargs)

        if mode == "direct":
            evolved = self.evolve(operations, **merged_device_kwargs)
            if output_type == "expval":
                return evolved.expval(observable)
            if output_type == "probs":
                return evolved.probs(wires=kwargs.get("wires", None))
            return evolved.sample(shots)

        if mode == "parallel":
            return self._execute_parallel(
                operations, observable, output_type, shots=shots, device_kwargs=merged_device_kwargs, **kwargs
            )
        return self._execute_sequential(
            operations, observable, output_type, shots=shots, device_kwargs=merged_device_kwargs, **kwargs
        )

    def expval(self, observable: Operator) -> TensorLike:
        r"""
        Analytic expectation value of an observable in the state.

        Pauli observables (single words, sums, Hamiltonians) are evaluated with the m-Pfaffian strategy from
        the extended covariance matrix, exactly as the device does for pure states. A
        :class:`~pennylane.Projector` on a basis state is evaluated as a probability, and a
        :class:`~matchcake.observables.BatchHamiltonian` term by term.

        :param observable: The observable.
        :type observable: Operator
        :return: The expectation value, of shape ``()`` or ``(B,)`` (``(n_terms, ...)`` for a
            ``BatchHamiltonian``).
        :rtype: TensorLike
        :raises NotImplementedError: If the observable is not supported.
        """
        if isinstance(observable, BatchHamiltonian):
            return qml.math.stack(
                [qml.math.real(coeff * self.expval(op)) for coeff, op in zip(observable.coeffs, observable.ops)]
            )
        if isinstance(observable, BasisStateProjector):
            return self._states_probability(observable.parameters[0], observable.wires)
        strategy = MPfaffianExpvalStrategy()
        reference_prep = self._reference_state_prep()
        if not strategy.can_execute(reference_prep, observable):
            raise NotImplementedError(
                f"The expectation value of {observable} cannot be computed from the covariance matrix of a "
                f"{type(self).__name__}. Use a Pauli observable, a basis-state Projector or a BatchHamiltonian."
            )
        return strategy(
            reference_prep,
            observable,
            extended_covariance_matrix=self.extended_covariance_matrix,
            pfaffian_chunk_size=self.device_kwargs.get("pfaffian_chunk_size", None),
        )

    def partial_trace(self, wires: WiresLike, **kwargs) -> "MixedGaussianState":
        r"""
        Trace out the given wires and return the state of the remaining ones.

        See :meth:`reduced_state` for the semantics; this method takes the wires to remove instead of the
        wires to keep.

        :param wires: Wires to trace out.
        :type wires: WiresLike
        :param kwargs: Forwarded to :meth:`reduced_state`.
        :return: The reduced state.
        :rtype: MixedGaussianState
        """
        traced = Wires(wires)
        missing = [wire for wire in traced.tolist() if wire not in self.wires]
        if missing:
            raise ValueError(f"Wires {missing} are not wires of this state ({self.wires.tolist()}).")
        kept = [wire for wire in self.wires.tolist() if wire not in traced]
        return self.reduced_state(kept, **kwargs)

    def probs(self, wires: Optional[WiresLike] = None) -> TensorLike:
        r"""
        Analytic probabilities of the computational-basis outcomes on the given wires.

        Each probability is :math:`2^{-k} |\mathrm{Pf}(\Lambda|_W + \Lambda_y)|`, the overlap of the
        Gaussian state with the basis state :math:`y` on the measured wires :math:`W`, as in
        :class:`~matchcake.devices.probability_strategies.ProductStateProbabilityStrategy`.

        :param wires: Measured wires, in the order of the returned outcomes. Defaults to all the wires.
        :type wires: Optional[WiresLike]
        :return: Probabilities of shape ``(2^k,)`` or ``(B, 2^k)``, with :math:`q_0` the most significant bit.
        :rtype: TensorLike
        """
        wires = self.wires if wires is None else Wires(wires)
        num_measured = len(wires)
        outcomes = NonInteractingFermionicDevice.states_to_binary(np.arange(2**num_measured), num_measured)
        probabilities = self._states_probability(outcomes, wires)  # (2^k, ...)
        probabilities = qml.math.moveaxis(probabilities, 0, -1)
        return probabilities / qml.math.sum(probabilities, axis=-1, keepdims=True)

    def pure_state_operations(self, index: int) -> List[Operation]:
        r"""
        Operations preparing the member ``index`` of the pure-state decomposition.

        Prepending them to a circuit on a :class:`~matchcake.devices.nif_device.NonInteractingFermionicDevice`
        with the wires of the state runs that circuit on :math:`V_R |x_\text{index}\rangle`.

        :param index: Index of the member, in ``range(num_pure_states)``.
        :type index: int
        :return: A :class:`~matchcake.operations.state_preparation.ProductState` followed by the
            :class:`~matchcake.operations.SingleParticleTransitionMatrixOperation` of :math:`R`.
        :rtype: List[Operation]
        """
        bits = self.basis_states[..., index, :]
        return [ProductState.from_basis_state(bits, wires=self.wires), self.sptm_operation]

    def pure_states(self) -> Iterator[Tuple[TensorLike, ProductState, SingleParticleTransitionMatrixOperation]]:
        r"""
        Iterate over the members of the pure-state decomposition.

        :return: Triples ``(weight, product_state, sptm_operation)``: the weight :math:`w_x` (of shape ``()``
            or ``(B,)``), the basis-state preparation :math:`|x\rangle` and the transition-matrix operation
            :math:`R`, such that the member is :math:`V_R |x\rangle`.
        :rtype: Iterator[Tuple[TensorLike, ProductState, SingleParticleTransitionMatrixOperation]]
        """
        sptm_operation = self.sptm_operation
        weights: Any = self.weights
        for index in range(self.num_pure_states):
            product_state, _ = self.pure_state_operations(index)
            yield weights[..., index], product_state, sptm_operation

    def reduced_state(self, wires: WiresLike, *, reduced_wires: Optional[WiresLike] = None) -> "MixedGaussianState":
        r"""
        Keep the given wires and trace out the others.

        The kept wires are ordered as in the state, and the reduced state is relabeled on ``reduced_wires``
        (``range(k)`` by default). The reduced covariance matrix is the principal submatrix of the covariance
        matrix on the Majorana indices of the kept wires, which is the fermionic partial trace: the Jordan-Wigner
        strings of the reduced system run through the kept wires only. It coincides with the qubit partial trace
        when the kept wires form a contiguous block.

        :param wires: Wires to keep.
        :type wires: WiresLike
        :param reduced_wires: Labels of the wires of the reduced state. Defaults to ``range(k)``.
        :type reduced_wires: Optional[WiresLike]
        :return: The reduced state.
        :rtype: MixedGaussianState
        """
        kept = Wires(wires)
        missing = [wire for wire in kept.tolist() if wire not in self.wires]
        if missing:
            raise ValueError(f"Wires {missing} are not wires of this state ({self.wires.tolist()}).")
        if len(kept) == 0:
            raise ValueError("At least one wire must be kept.")
        positions = sorted(self.wires.indices(kept))
        indices = self.majorana_indices(positions)
        covariance_matrix: Any = self.covariance_matrix
        covariance = covariance_matrix[..., indices[:, None], indices[None, :]]
        source_wires = [self.source_wires.tolist()[position] for position in positions]
        return self._new(covariance, wires=reduced_wires, source_wires=source_wires)

    def sample(self, shots: Optional[int], sampling_strategy: Optional[str] = None) -> np.ndarray:
        r"""
        Draw computational-basis samples from the state.

        The samples are generated wire block by wire block with the chain rule of probability, using the
        sampling strategies of the device with the probabilities of :meth:`probs`.

        :param shots: Number of samples.
        :type shots: Optional[int]
        :param sampling_strategy: Name of the sampling strategy. Defaults to the device default.
        :type sampling_strategy: Optional[str]
        :return: Integer samples of shape ``(shots, m)`` or ``(shots, B, m)``.
        :rtype: np.ndarray
        """
        if shots is None:
            raise ValueError("The number of shots must be specified to generate samples.")
        device = self._make_device(shots=shots)
        strategy = get_sampling_strategy(sampling_strategy or self.DEFAULT_SAMPLING_STRATEGY)
        return np.asarray(
            strategy.batch_generate_samples(device, self._states_probability, show_progress=False)
        ).astype(int)

    def _check_operations(self, operations: Iterable[Operation]) -> List[Operation]:
        """Materialize the operations and reject state preparations and foreign wires.

        :param operations: The operations of a circuit.
        :type operations: Iterable[Operation]
        :return: The operations as a list.
        :rtype: List[Operation]
        """
        operations = list(operations)
        for operation in operations:
            if isinstance(operation, StatePrepBase):
                raise ValueError(
                    f"The circuit run on a {type(self).__name__} cannot contain a state preparation, "
                    f"got {operation}. The state itself is the initial state."
                )
            foreign = [wire for wire in operation.wires.tolist() if wire not in self.wires]
            if foreign:
                raise ValueError(
                    f"Operation {operation} acts on wires {foreign} which are not wires of the state "
                    f"({self.wires.tolist()})."
                )
        return operations

    def _execute_parallel(
        self,
        operations: List[Operation],
        observable: Optional[Operator],
        output_type: OutputType,
        *,
        shots: Optional[int],
        device_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Run the circuit once on all the members at once and recombine the outputs.

        :return: The recombined output.
        :rtype: Any
        """
        device = self._make_device(shots=shots, **device_kwargs)
        result = device.execute_generator(
            self.ensemble_operations() + operations,
            observable=observable,
            output_type=output_type,
            reset=True,
            **kwargs,
        )
        num_members = self.num_pure_states
        ensemble_shape = self.batch_shape + (num_members,)
        if output_type == "samples":
            samples = np.asarray(result)  # (shots, N, m)
            samples = samples.reshape(samples.shape[:1] + ensemble_shape + samples.shape[-1:])
            return self._select_members(samples)
        weights = self.weights
        if output_type == "probs":
            result = qml.math.reshape(result, ensemble_shape + tuple(qml.math.shape(result))[1:])
            weights = convert_and_cast_like(weights, result)
            return qml.math.sum(weights[..., None] * result, axis=-2)
        if isinstance(observable, BatchHamiltonian):
            result = qml.math.reshape(result, tuple(qml.math.shape(result))[:1] + ensemble_shape)
        else:
            result = qml.math.reshape(result, ensemble_shape)
        weights = convert_and_cast_like(weights, result)
        return qml.math.sum(weights * result, axis=-1)

    def _execute_sequential(
        self,
        operations: List[Operation],
        observable: Optional[Operator],
        output_type: OutputType,
        *,
        shots: Optional[int],
        device_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Run the circuit on the members one after the other and recombine the outputs.

        :return: The recombined output.
        :rtype: Any
        """
        device = self._make_device(shots=shots, **device_kwargs)
        results = [
            device.execute_generator(
                self.pure_state_operations(index) + operations,
                observable=observable,
                output_type=output_type,
                reset=True,
                **kwargs,
            )
            for index in range(self.num_pure_states)
        ]
        if output_type == "samples":
            samples = np.stack([np.asarray(sample) for sample in results], axis=-2)  # (shots, *batch, K, m)
            return self._select_members(samples)
        weights = convert_and_cast_like(self.weights, results[0])
        total = None
        for index, result in enumerate(results):
            weight = weights[..., index]
            if output_type == "probs":
                weight = qml.math.reshape(weight, tuple(qml.math.shape(weight)) + (1,))
            term = weight * result
            total = term if total is None else total + term
        return total

    def _make_device(self, **device_kwargs) -> NonInteractingFermionicDevice:
        """Create a device on the wires of the state.

        :param device_kwargs: Overrides of the device keyword arguments of the state.
        :return: The device.
        :rtype: NonInteractingFermionicDevice
        """
        return NonInteractingFermionicDevice(wires=self.wires, **{**self.device_kwargs, **device_kwargs})

    def _new(
        self, covariance_matrix: TensorLike, wires: Optional[WiresLike], source_wires: Optional[WiresLike]
    ) -> "MixedGaussianState":
        """Build a state sharing the settings of this one.

        :param covariance_matrix: Covariance matrix of the new state.
        :type covariance_matrix: TensorLike
        :param wires: Wires of the new state.
        :type wires: Optional[WiresLike]
        :param source_wires: Source wires of the new state.
        :type source_wires: Optional[WiresLike]
        :return: The new state.
        :rtype: MixedGaussianState
        """
        return type(self)(
            covariance_matrix,
            wires=wires,
            atol=self.atol,
            device_kwargs=self.device_kwargs,
            source_wires=source_wires,
        )

    def _process_samples(
        self,
        samples: np.ndarray,
        *,
        observable: Optional[Operator],
        output_type: OutputType,
        **kwargs,
    ) -> TensorLike:
        """Estimate an expectation value or probabilities from samples of the state.

        :param samples: Samples of shape ``(shots, m)`` or ``(shots, B, m)``.
        :type samples: np.ndarray
        :return: The estimate, with the batch of the state leading.
        :rtype: TensorLike
        """
        if output_type == "expval":
            measurement: Any = ExpectationMP(obs=observable)
        else:
            measurement = ProbabilityMP(wires=Wires(kwargs.get("wires", None) or self.wires))
        flat_samples = samples.reshape(samples.shape[0], -1, samples.shape[-1])  # (shots, prod(batch), m)
        estimates = [
            measurement.process_samples(
                flat_samples[:, batch_index, :],
                wire_order=self.wires,
                shot_range=kwargs.get("shot_range", None),
                bin_size=kwargs.get("bin_size", None),
            )
            for batch_index in range(flat_samples.shape[1])
        ]
        if len(self.batch_shape) == 0:
            return estimates[0]
        return qml.math.stack(estimates, axis=0)

    def _reference_state_prep(self) -> ProductState:
        """Return a basis-state preparation on the wires, used to route the strategies of the device.

        :return: The preparation of the all-zero basis state.
        :rtype: ProductState
        """
        return ProductState.from_basis_state(np.zeros(self.num_wires, dtype=int), wires=self.wires)

    def _select_members(self, samples: np.ndarray) -> np.ndarray:
        """Draw the member of every shot from the weights and keep that member's sample.

        :param samples: Samples of every member, of shape ``(shots, *batch, K, m)``.
        :type samples: np.ndarray
        :return: Samples of the mixture, of shape ``(shots, *batch, m)``.
        :rtype: np.ndarray
        """
        shots = samples.shape[0]
        member_index = random_index(detach(self.weights), n=shots, axis=-1)  # (shots, *batch)
        member_index = np.asarray(member_index, dtype=int).reshape(samples.shape[:-2])
        selected = np.take_along_axis(samples, member_index[..., None, None], axis=-2)
        return selected[..., 0, :]

    def _states_probability(self, target_binary_states: TensorLike, wires: WiresLike) -> TensorLike:
        """Probabilities of basis outcomes on the given wires, from the covariance matrix.

        :param target_binary_states: Outcomes of shape ``(k,)`` or ``(n_outcomes, k)``.
        :type target_binary_states: TensorLike
        :param wires: Measured wires.
        :type wires: WiresLike
        :return: Probabilities of shape ``(...)`` or ``(n_outcomes, ...)``.
        :rtype: TensorLike
        """
        strategy = ProductStateProbabilityStrategy()
        return strategy(
            state_prep_op=self._reference_state_prep(),
            target_binary_states=np.asarray(target_binary_states, dtype=int),
            wires=np.asarray(Wires(wires).tolist()),
            covariance_matrix=self.covariance_matrix,
            all_wires=self.wires,
            pfaffian_chunk_size=self.device_kwargs.get("pfaffian_chunk_size", None),
        )

    def _mixed_combinations(self) -> np.ndarray:
        """Every assignment of the bits of the mixed modes, in lexicographic order.

        :return: Integer array of shape ``(K, r)``.
        :rtype: np.ndarray
        """
        num_mixed = self.num_mixed_modes
        return np.array(list(itertools.product([0, 1], repeat=num_mixed)), dtype=int).reshape(2**num_mixed, num_mixed)

    @cached_property
    def basis_states(self) -> np.ndarray:
        r"""
        Basis states :math:`x` of the members of the pure-state decomposition.

        Pure modes carry the bit :math:`x_j = 0` when :math:`\lambda_j > 0` and :math:`x_j = 1` otherwise,
        while the mixed modes run over every combination, in lexicographic order.

        :return: Integer array of shape ``(K, m)`` or ``(B, K, m)``.
        :rtype: np.ndarray
        """
        polarizations = np.asarray(detach(self.mode_polarizations))
        pure_bits = (polarizations < 0).astype(int)  # (..., m)
        mixed_modes = self.mixed_modes
        combinations = self._mixed_combinations()
        basis = np.broadcast_to(pure_bits[..., None, :], self.batch_shape + combinations.shape[:1] + (self.num_wires,))
        basis = np.array(basis, dtype=int)
        basis[..., :, mixed_modes] = combinations
        return basis

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """The batch shape of the covariance matrix, ``()`` or ``(B,)``.

        :return: The batch shape.
        :rtype: Tuple[int, ...]
        """
        return tuple(qml.math.shape(self._covariance_matrix))[:-2]

    @property
    def covariance_matrix(self) -> TensorLike:
        """The Majorana covariance matrix of the state, of shape ``(..., 2m, 2m)``.

        :return: The covariance matrix.
        :rtype: TensorLike
        """
        return self._covariance_matrix

    @cached_property
    def extended_covariance_matrix(self) -> TensorLike:
        """The covariance matrix bordered by a zero displacement, of shape ``(..., 2m+1, 2m+1)``.

        The state has a definite parity, so the parity row and column of the extended covariance matrix of
        the m-Pfaffian strategy vanish.

        :return: The extended covariance matrix.
        :rtype: TensorLike
        """
        covariance = self.covariance_matrix
        displacement = convert_and_cast_like(np.zeros(self.batch_shape + (2 * self.num_wires,)), covariance)
        return extended_covariance_matrix(covariance, displacement)

    @property
    def is_pure(self) -> TensorLike:
        """Whether every mode is pure, within ``atol``.

        :return: A boolean, or a boolean array of shape ``(B,)``.
        :rtype: TensorLike
        """
        polarizations = np.asarray(detach(self.mode_polarizations))
        return np.all(np.abs(polarizations) >= 1.0 - self.atol, axis=-1)

    @cached_property
    def mixed_modes(self) -> List[int]:
        """Indices of the normal modes that are mixed in at least one element of the batch.

        :return: Sorted mode indices.
        :rtype: List[int]
        """
        polarizations = np.asarray(detach(self.mode_polarizations)).reshape(-1, self.num_wires)
        is_mixed = np.any(np.abs(polarizations) < 1.0 - self.atol, axis=0)
        return [int(index) for index in np.flatnonzero(is_mixed)]

    @property
    def mode_polarizations(self) -> TensorLike:
        r"""
        Polarizations :math:`\lambda_j = \langle Z_j \rangle \in [-1, 1]` of the normal modes.

        :return: Polarizations of shape ``(m,)`` or ``(B, m)``.
        :rtype: TensorLike
        """
        return self._decomposition[1]

    @property
    def num_mixed_modes(self) -> int:
        """The number of mixed normal modes.

        :return: The number of mixed modes.
        :rtype: int
        """
        return len(self.mixed_modes)

    @property
    def num_pure_states(self) -> int:
        """The number :math:`K = 2^r` of members of the pure-state decomposition.

        :return: The number of members.
        :rtype: int
        """
        return 2**self.num_mixed_modes

    @property
    def num_wires(self) -> int:
        """The number of wires of the state.

        :return: The number of wires.
        :rtype: int
        """
        return len(self._wires)

    @property
    def purity(self) -> TensorLike:
        r"""
        Purity :math:`\mathrm{Tr}[\rho^2] = \prod_j \tfrac{1 + \lambda_j^2}{2}` of the state.

        :return: The purity, of shape ``()`` or ``(B,)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        return qml.math.prod(0.5 * (1.0 + polarizations**2), axis=-1)

    @property
    def source_wires(self) -> Wires:
        """The labels, in the original system, of the wires this state was reduced to.

        :return: The source wires.
        :rtype: Wires
        """
        return self._source_wires

    @property
    def sptm(self) -> TensorLike:
        r"""
        Single-particle transition matrix :math:`R \in SO(2m)` of the normal-mode decomposition.

        :return: Matrix of shape ``(2m, 2m)`` or ``(B, 2m, 2m)``.
        :rtype: TensorLike
        """
        return self._decomposition[0]

    @property
    def sptm_operation(self) -> SingleParticleTransitionMatrixOperation:
        """The transition matrix :math:`R` as an operation on the wires of the state.

        :return: The operation.
        :rtype: SingleParticleTransitionMatrixOperation
        """
        return SingleParticleTransitionMatrixOperation(self.sptm, wires=self.wires)

    @cached_property
    def weights(self) -> TensorLike:
        r"""
        Weights :math:`w_x` of the members of the pure-state decomposition, summing to one.

        A pure mode contributes a factor one (its polarization is treated as exactly :math:`\pm 1`), and
        each mixed mode contributes :math:`(1 + (-1)^{x_j} \lambda_j) / 2`. The weights are differentiable
        with respect to the covariance matrix through the polarizations.

        :return: Weights of shape ``(K,)`` or ``(B, K)``.
        :rtype: TensorLike
        """
        polarizations: Any = self.mode_polarizations
        mixed_modes = self.mixed_modes
        combinations = self._mixed_combinations()  # (K, r)
        if len(mixed_modes) == 0:
            return convert_and_cast_like(np.ones(self.batch_shape + (1,)), polarizations)
        signs = convert_and_cast_like(1 - 2 * combinations, polarizations)  # (K, r)
        mixed_polarizations = polarizations[..., mixed_modes]  # (..., r)
        factors = 0.5 * (1.0 + signs * mixed_polarizations[..., None, :])  # (..., K, r)
        return qml.math.prod(factors, axis=-1)

    @property
    def wires(self) -> Wires:
        """The wires of the state.

        :return: The wires.
        :rtype: Wires
        """
        return self._wires

    @cached_property
    def _decomposition(self) -> Tuple[TensorLike, TensorLike]:
        """The block-diagonalization ``(R, polarizations)`` of the covariance matrix.

        :return: The transition matrix and the polarizations.
        :rtype: Tuple[TensorLike, TensorLike]
        """
        return block_diagonalize_covariance(self.covariance_matrix)
