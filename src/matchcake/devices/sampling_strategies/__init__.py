from typing import Union

from ...utils import get_all_subclasses
from .product_basis_sampling import ProductBasisSampling
from .qubit_by_qubit_sampling import QubitByQubitSampling
from .sampling_strategy import SamplingStrategy
from .two_qubits_by_two_qubits_sampling import TwoQubitsByTwoQubitsSampling

sampling_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(SamplingStrategy)}


def get_sampling_strategy(name: Union[str, SamplingStrategy]) -> SamplingStrategy:
    """Return a sampling strategy from its name, or the given strategy itself.

    :param name: The name of a sampling strategy, case-insensitive, or a strategy instance.
    :type name: Union[str, SamplingStrategy]
    :return: A new strategy of the given name, or ``name`` when it is already a strategy.
    :rtype: SamplingStrategy
    :raises ValueError: If no strategy has this name.
    """
    if isinstance(name, SamplingStrategy):
        return name
    ansatz_name = name.lower().strip()
    if ansatz_name not in sampling_strategy_map:
        raise ValueError(f"Unknown sampling strategy name: {ansatz_name}")
    return sampling_strategy_map[ansatz_name]()
