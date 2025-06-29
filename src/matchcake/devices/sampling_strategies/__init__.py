from ...utils import get_all_subclasses
from .qubit_by_qubit_sampling import QubitByQubitSampling
from .sampling_strategy import SamplingStrategy
from .two_qubits_by_two_qubits_sampling import TwoQubitsByTwoQubitsSampling

sampling_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(SamplingStrategy)}


def get_sampling_strategy(name: str) -> SamplingStrategy:
    ansatz_name = name.lower().strip()
    if ansatz_name not in sampling_strategy_map:
        raise ValueError(f"Unknown sampling strategy name: {ansatz_name}")
    return sampling_strategy_map[ansatz_name]()
