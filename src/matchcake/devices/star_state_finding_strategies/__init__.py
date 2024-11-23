from .star_state_finding_strategy import StarStateFindingStrategy
from ...utils import get_all_subclasses

star_state_finding_strategy_map = {
    _cls.NAME.lower().strip(): _cls
    for _cls in get_all_subclasses(StarStateFindingStrategy)
}


def get_star_state_finding_strategy(name: str) -> StarStateFindingStrategy:
    name = name.lower().strip()
    if name not in star_state_finding_strategy_map:
        raise ValueError(f"Unknown star state finding strategy name: {name}")
    return star_state_finding_strategy_map[name]()
