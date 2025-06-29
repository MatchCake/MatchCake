from typing import Optional

from ...utils import get_all_subclasses
from .linear_strategy import *
from .parameters_initialisation_strategy import ParametersInitialisationStrategy
from .random_strategy import *

parameters_initialisation_strategy_map = {
    _cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(ParametersInitialisationStrategy)
}


def get_parameters_initialisation_strategy(
    name: Optional[str],
) -> ParametersInitialisationStrategy:
    name = str(name).lower().strip()
    if name not in parameters_initialisation_strategy_map:
        raise ValueError(
            f"Unknown {ParametersInitialisationStrategy.NAME} name: {name}. "
            f"Available strategies: {list(parameters_initialisation_strategy_map.keys())}"
        )
    return parameters_initialisation_strategy_map[name]()
