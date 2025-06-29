from typing import Union

from ...utils import get_all_subclasses
from .from_sampling_strategy import FromSamplingStrategy
from .greedy_strategy import GreedyStrategy
from .star_state_finding_strategy import StarStateFindingStrategy

star_state_finding_strategy_map = {
    _cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(StarStateFindingStrategy)
}


def get_star_state_finding_strategy(
    name: Union[str, StarStateFindingStrategy],
) -> StarStateFindingStrategy:
    if isinstance(name, StarStateFindingStrategy):
        return name
    name = name.lower().strip()
    if name not in star_state_finding_strategy_map:
        raise ValueError(f"Unknown star state finding strategy name: {name}")
    return star_state_finding_strategy_map[name]()
