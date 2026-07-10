from abc import ABC
from typing import List

import numpy as np
import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.ops.qubit.state_preparation import BasisState
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pythonbasictools.multiprocessing_tools import apply_func_multiprocess

from matchcake.operations.state_preparation import StatePrepFromGates
from matchcake.operations.state_preparation.product_state import ProductState


class ProbabilityStrategy(ABC):
    NAME: str = "ProbabilityStrategy"
    REQUIRES_KWARGS: List[str] = []

    @classmethod
    def system_basis_state_from_state_prep_op(cls, state_prep_op: StatePrepBase) -> TensorLike:
        """Extract the computational-basis bit string from a state-preparation op.

        :param state_prep_op: The state preparation operation.
        :type state_prep_op: StatePrepBase
        :return: Integer bit array of shape ``(n,)``.
        :rtype: TensorLike
        """
        if isinstance(state_prep_op, BasisState):
            return state_prep_op.parameters[0]
        elif isinstance(state_prep_op, StatePrepFromGates):
            return state_prep_op.to_basis_state().parameters[0]
        elif isinstance(state_prep_op, ProductState):
            return state_prep_op.as_basis_state().parameters[0]
        else:
            raise NotImplementedError(f"{cls.__name__} cannot be used with {type(state_prep_op)}")

    def __call__(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_states: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        """Compute probabilities for one or a batch of basis outcomes.

        Detects the input dimensionality:

        * 1-D input ``(k,)`` — delegates to :meth:`_compute_single` and returns a scalar.
        * 2-D input ``(B, k)`` — loops over rows via ``apply_func_multiprocess`` and
          returns a ``(B,)`` vector.

        Subclasses may override this method entirely to implement an efficient batched
        computation without calling :meth:`_compute_single`.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :param target_binary_states: Binary outcomes of shape ``(k,)`` or ``(B, k)``.
        :type target_binary_states: TensorLike
        :param wires: Measured wires — a flat ``Wires`` for single, or ``ndarray(B, k)``
            (one row per outcome) for batch.
        :type wires: Wires
        :return: Scalar probability for 1-D input, ``(B,)`` vector for 2-D input.
        :rtype: TensorLike
        """
        target_arr = np.asarray(target_binary_states)
        is_single = target_arr.ndim == 1

        if is_single:
            return self._compute_single(
                state_prep_op=state_prep_op,
                target_binary_state=target_arr,
                wires=Wires(wires),
                **kwargs,
            )

        wires_arr = np.asarray(wires)
        if wires_arr.ndim == 1:
            wires_arr = np.broadcast_to(wires_arr, target_arr.shape)

        show_progress = kwargs.pop("show_progress", False)
        nb_workers = kwargs.pop("nb_workers", 0)
        probs_list = apply_func_multiprocess(
            func=self,
            iterable_of_args=[() for _ in range(len(target_arr))],
            iterable_of_kwargs=[
                {
                    "state_prep_op": state_prep_op,
                    "target_binary_states": tbs,
                    "wires": Wires(w),
                    **kwargs,
                }
                for tbs, w in zip(target_arr, wires_arr)
            ],
            verbose=show_progress,
            desc=f"[{self.NAME}] Batch Probability Calculation with {nb_workers} workers",
            unit="states",
            nb_workers=nb_workers,
        )
        return qml.math.stack(probs_list, axis=0)

    def check_required_kwargs(self, kwargs: dict) -> "ProbabilityStrategy":
        """Raise ``ValueError`` if any entry of :attr:`REQUIRES_KWARGS` is absent.

        :param kwargs: Keyword arguments to validate.
        :type kwargs: dict
        :return: Self, for chaining.
        :rtype: ProbabilityStrategy
        """
        for kwarg in self.REQUIRES_KWARGS:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required keyword argument: {kwarg}")
        return self

    def can_execute(self, state_prep_op: StatePrepBase) -> bool:
        """Return whether this strategy can handle the given state preparation.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :return: True if this strategy can compute the probability for ``state_prep_op``.
        :rtype: bool
        """
        return True

    def _compute_single(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_state: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        """Compute the probability for a single basis outcome.

        The default implementation raises :class:`NotImplementedError`. Subclasses that
        do not override :meth:`__call__` must implement this method.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :param target_binary_state: Binary outcome of shape ``(k,)``.
        :type target_binary_state: TensorLike
        :param wires: Measured wires.
        :type wires: Wires
        :return: Scalar probability.
        :rtype: TensorLike
        """
        raise NotImplementedError(f"{type(self).__name__} must implement either __call__ or _compute_single.")
