import importlib
from typing import Any, List, Callable, Union
import functools

import numpy as np


def recursive_kron(__inputs: List[Any], lib=np, recursive: bool = True) -> Any:
    if isinstance(lib, str):
        lib = importlib.import_module(lib)
    return recursive_2in_operator(lib.kron, __inputs, recursive=recursive)


def recursive_2in_operator(
        operator: Callable[[Any, Any], Any],
        __inputs: List[Any],
        recursive: bool = True,
) -> Any:
    r"""
    Apply an operator recursively to a list of inputs. The operator must accept two inputs. The inputs are applied
    from left to right.

    # TODO: try to go from left to right and from right to left and compare the performance.

    :param operator: Operator to apply
    :type operator: Callable[[Any, Any], Any]
    :param __inputs: Inputs to apply the operator to
    :type __inputs: List[Any]
    :param recursive: If True, apply the operator recursively. If False, apply the operator iteratively using
        functools.reduce.
    :type recursive: bool
    :return: Result of the operator applied to the inputs
    :rtype: Any
    """
    if len(__inputs) == 1:
        return __inputs[0]
    elif len(__inputs) == 2:
        return operator(__inputs[0], __inputs[1])
    elif len(__inputs) > 2:
        if recursive:
            rec = recursive_2in_operator(operator, __inputs[:-1])
            return operator(rec, __inputs[-1])
        else:
            return functools.reduce(operator, __inputs)
    else:
        raise ValueError("Invalid shape for input array")





