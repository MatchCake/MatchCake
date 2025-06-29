from typing import Optional

from ...utils import get_all_subclasses
from .contraction_strategy import ContractionStrategy
from .forward_strategy import ForwardContractionStrategy
from .horizontal_strategy import HorizontalContractionStrategy
from .neighbours_strategy import NeighboursContractionStrategy
from .none_strategy import NoneContractionStrategy
from .vertical_strategy import VerticalContractionStrategy

contraction_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(ContractionStrategy)}


def get_contraction_strategy(name: Optional[str]) -> ContractionStrategy:
    name = str(name).lower().strip()
    if name not in contraction_strategy_map:
        raise ValueError(
            f"Unknown contraction strategy name: {name}. "
            f"Available strategies: {list(contraction_strategy_map.keys())}"
        )
    return contraction_strategy_map[name]()
