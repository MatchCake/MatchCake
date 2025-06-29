import functools
import importlib
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
from pennylane.wires import Wires

from .constants import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
from .operators import recursive_2in_operator, recursive_kron


def get_majorana_pauli_list(i: int, n: int) -> List[np.ndarray]:
    r"""

    Get the list of Pauli matrices for the computation of the Majorana operator :math:`c_i` defined as

    .. math::
        c_{2k+1} = Z^{\otimes k} \otimes X \otimes I^{\otimes n-k-1}

    for odd :math:`i` and

    .. math::
        c_{2k} = Z^{\otimes k} \otimes Y \otimes I^{\otimes n-k-1}

    for even :math:`i`, where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`X`
    is the Pauli X matrix, :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana
    operator and :math:`n` is the number of particles.

    :param i: Index of the Majorana operator
    :type i: int
    :param n: Number of particles
    :type n: int
    :return: List of Pauli matrices
    :rtype: List[np.ndarray]
    """
    assert 0 <= i < 2 * n
    k = int(i / 2)  # 0, ..., n-1
    if (i + 1) % 2 == 0:
        gate = PAULI_Y
    else:
        gate = PAULI_X
    # return [PAULI_Z] * k + [gate] + [PAULI_I] * (n - k - 1)
    return [PAULI_Z for _ in range(k)] + [gate] + [PAULI_I for _ in range(n - k - 1)]


def get_majorana_pauli_string(i: int, n: int, join_char="⊗") -> str:
    assert 0 <= i < 2 * n
    k = int(i / 2)  # 0, ..., n-1
    if (i + 1) % 2 == 0:
        gate = "Y"
    else:
        gate = "X"
    return join_char.join(["Z"] * k + [gate] + ["I"] * (n - k - 1))


@functools.lru_cache(maxsize=128)
def get_majorana(i: int, n: int) -> np.ndarray:
    r"""
    Get the Majorana matrix defined as

    .. math::
        c_{2k+1} = Z^{\otimes k} \otimes X \otimes I^{\otimes n-k-1}

    for odd :math:`i` and

    .. math::
        c_{2k} = Z^{\otimes k} \otimes Y \otimes I^{\otimes n-k-1}

    for even :math:`i`, where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix, :math:`X`
    is the Pauli X matrix, :math:`\otimes` is the Kronecker product, :math:`k` is the index of the Majorana
    operator and :math:`n` is the number of particles.

    :Note: The index :math:`i` starts from 0.

    :param i: Index of the Majorana operator
    :type i: int
    :param n: Number of particles
    :type n: int
    :return: Majorana matrix
    :rtype: np.ndarray
    """
    return recursive_kron(get_majorana_pauli_list(i, n))


@functools.lru_cache(maxsize=128)
def get_majorana_pair(i: int, j: int, n: int) -> np.ndarray:
    r"""
    Get the Majorana pair defined as

    .. math::
        c_{2k+1} c_{2l+1} = Z^{\otimes k+l} \otimes X \otimes I^{\otimes n-k-l-1}

    for odd :math:`i` and :math:`j` and

    .. math::
        c_{2k} c_{2l} = Z^{\otimes k+l} \otimes Y \otimes I^{\otimes n-k-l-1}

    for even :math:`i` and :math:`j`, where :math:`Z` is the Pauli Z matrix, :math:`I` is the identity matrix,
    :math:`X` is the Pauli X matrix, :math:`\otimes` is the Kronecker product, :math:`k` and :math:`l` are the
    indices of the Majorana operators and :math:`n` is the number of particles.

    :Note: The indices :math:`i` and :math:`j` start from 0.

    :param i: Index of the first Majorana operator
    :type i: int
    :param j: Index of the second Majorana operator
    :type j: int
    :param n: Number of particles
    :type n: int
    :return: Majorana pair
    :rtype: np.ndarray
    """
    assert 0 <= i < 2 * n
    assert 0 <= j < 2 * n
    if i == j:
        return np.eye(2**n)
    return get_majorana(i, n) @ get_majorana(j, n)


class MajoranaGetter:
    r"""

    Class for caching the Majorana matrices. The Majorana matrices are computed using the function
    :func:`get_majorana`. The matrices are cached in a dictionary for faster computation.

    :param n: Number of particles
    :type n: int

    """

    def __init__(self, n: int, maxsize=None):
        self.n = n
        self.maxsize = maxsize or np.inf
        self.cache = OrderedDict()

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, item: Union[int, Tuple[int, int]]) -> np.ndarray:
        if isinstance(item, tuple):
            i, j = item
        else:
            i, j = item, None
        if j is None:
            return self._get_or_compute_majorana(i)
        else:
            return self._get_or_compute_majorana_pair(i, j)

    def _get_or_compute_majorana(self, i: int) -> np.ndarray:
        if i not in self.cache:
            self.cache[i] = get_majorana(i, self.n)
        return self.cache[i]

    def _get_or_compute_majorana_pair(self, i: int, j: int) -> np.ndarray:
        if (i, j) not in self.cache:
            if (j, i) in self.cache:
                return -self.cache[(j, i)]
            else:
                self.cache[(i, j)] = get_majorana_pair(i, j, self.n)
        return self.cache[(i, j)]

    def cache_item(self, key: Any, value: Any) -> Any:
        r"""
        Cache an item. If the cache is full, the oldest item is removed.

        :param key: The key of the item.
        :param value: The value of the item.
        :return: The removed item.
        """
        removed = None
        if self.maxsize is not None and len(self.cache) >= self.maxsize:
            removed = self.cache.popitem(last=False)
        self.cache[key] = value
        return removed

    def __setitem__(self, key: Any, value: Any):
        self.cache_item(key, value)

    def __call__(self, i: int, j: Optional[int] = None) -> np.ndarray:
        return self[i, j]

    def __len__(self) -> int:
        r"""
        Return the number of Majorana matrices for the given number of particles plus the number of
        Majorana pairs.

        :return: The number of Majorana matrices.
        """
        return self.n * 2 + self.n * (self.n - 1) // 2

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def clear_cache(self):
        self.cache = {}
