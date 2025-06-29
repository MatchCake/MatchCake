from typing import Optional, Union

from ...utils import get_all_subclasses
from .adam_strategy import *
from .adamw_strategy import *
from .genetic_strategy import *
from .optimizer_strategy import OptimizerStrategy
from .random_strategy import *
from .scipy_strategies import *
from .simulated_annealing_strategy import *

# from .fqaoa_simulated_annealing_strategy import *
# from .grid_search_strategy import *

optimizer_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(OptimizerStrategy)}


def get_optimizer_strategy(
    name: Optional[Union[str, OptimizerStrategy]],
) -> OptimizerStrategy:
    if isinstance(name, OptimizerStrategy):
        return name
    name = str(name).lower().strip()
    if name not in optimizer_strategy_map:
        raise ValueError(
            f"Unknown {OptimizerStrategy.NAME} name: {name}. "
            f"Available strategies: {list(optimizer_strategy_map.keys())}"
        )
    return optimizer_strategy_map[name]()
