from typing import Optional, Union

from ...utils import get_all_subclasses
from .bulk_strategy import BulkContractionStrategy
from .contraction_strategy import ContractionStrategy
from .forward_strategy import ForwardContractionStrategy
from .horizontal_strategy import HorizontalContractionStrategy
from .neighbours_strategy import NeighboursContractionStrategy
from .none_strategy import NoneContractionStrategy
from .vertical_strategy import VerticalContractionStrategy

contraction_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(ContractionStrategy)}


def get_contraction_strategy(name: Optional[Union[str, ContractionStrategy]]) -> ContractionStrategy:
    """Return a contraction strategy from its name, or the strategy itself.

    :param name: The name of a strategy, any case, ``None`` for the strategy that contracts nothing,
        or a strategy instance, which is returned as it is.
    :type name: Optional[Union[str, ContractionStrategy]]
    :return: The strategy.
    :rtype: ContractionStrategy
    :raises ValueError: If the name is not that of a strategy.
    """
    if isinstance(name, ContractionStrategy):
        return name
    name = str(name).lower().strip()
    if name not in contraction_strategy_map:
        raise ValueError(
            f"Unknown contraction strategy name: {name}. Available strategies: {list(contraction_strategy_map.keys())}"
        )
    return contraction_strategy_map[name]()
