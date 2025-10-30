from ...utils import get_all_subclasses
from .explicit_sum_strategy import ExplicitSumStrategy
from .lookup_table_strategy import LookupTableStrategy
from .probability_strategy import ProbabilityStrategy

probability_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(ProbabilityStrategy)}


def get_probability_strategy(name: str) -> ProbabilityStrategy:
    name = name.lower().strip()
    if name not in probability_strategy_map:
        raise ValueError(
            f"Unknown sampling strategy name: {name}. " f"Available strategies: {list(probability_strategy_map.keys())}"
        )
    return probability_strategy_map[name]()
