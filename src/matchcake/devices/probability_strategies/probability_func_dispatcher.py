from typing import List

from pennylane.operation import StatePrepBase
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .probability_strategy import ProbabilityStrategy


class ProbabilityFuncDispatcher:
    """Dispatch a probability computation to the first compatible strategy.

    Strategies are evaluated in order; the first one whose :meth:`can_execute` returns
    ``True`` for the current :attr:`state_prep_op` is used to compute the result.

    This mirrors the role of
    :class:`~matchcake.devices.expval_strategies.terms_splitter.TermsSplitter` on the
    expval side: it decouples the routing logic from both the device and the individual
    strategies, and makes it easy to extend the set of supported state preparations
    by appending a new strategy.
    """

    NAME: str = "ProbabilityFuncDispatcher"

    def __init__(self, strategies: List[ProbabilityStrategy]) -> None:
        """
        :param strategies: Ordered list of probability strategies. The first strategy
            whose :meth:`~ProbabilityStrategy.can_execute` returns ``True`` is used.
        :type strategies: List[ProbabilityStrategy]
        """
        self._strategies = strategies

    def __call__(
        self,
        *,
        state_prep_op: StatePrepBase,
        target_binary_states: TensorLike,
        wires: Wires,
        **kwargs,
    ) -> TensorLike:
        """Compute probabilities by delegating to the first compatible strategy.

        :param state_prep_op: State preparation operation used to select the strategy.
        :type state_prep_op: StatePrepBase
        :param target_binary_states: Binary outcomes of shape ``(k,)`` or ``(B, k)``.
        :type target_binary_states: TensorLike
        :param wires: Measured wires.
        :type wires: Wires
        :return: Scalar or ``(B,)`` probabilities from the selected strategy.
        :rtype: TensorLike
        :raises ValueError: If no strategy in the list can handle ``state_prep_op``.
        """
        for strategy in self._strategies:
            if strategy.can_execute(state_prep_op):
                return strategy(
                    state_prep_op=state_prep_op,
                    target_binary_states=target_binary_states,
                    wires=wires,
                    **kwargs,
                )
        strategy_names = [s.NAME for s in self._strategies]
        raise ValueError(
            f"No probability strategy can execute for state prep op of type "
            f"'{type(state_prep_op).__name__}'. "
            f"Available strategies: {strategy_names}."
        )

    def can_execute(self, state_prep_op: StatePrepBase) -> bool:
        """Return True if at least one strategy in the list can handle this state.

        :param state_prep_op: State preparation operation.
        :type state_prep_op: StatePrepBase
        :return: True when at least one contained strategy can execute.
        :rtype: bool
        """
        return any(s.can_execute(state_prep_op) for s in self._strategies)

    @property
    def strategies(self) -> List[ProbabilityStrategy]:
        """The ordered list of probability strategies.

        :return: List of strategies in dispatch order.
        :rtype: List[ProbabilityStrategy]
        """
        return self._strategies
