import itertools
import numbers
from functools import cached_property
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
import tqdm
from pennylane import numpy as pnp
from pennylane.typing import TensorLike
from scipy import sparse

from .. import utils


class NonInteractingFermionicLookupTable:
    r"""
    Represents a lookup table for computations related to non-interacting fermionic systems.

    This class is designed to handle the Lookup table describe in :cite:`10821385` . More specifically, it is design to
    find the tensor that represents the contraction of the summation

    .. math::

        p(y^{*}|x) &= \prod_{\gamma = 1}^{k}\sum_{m_{\gamma},n_{\gamma} = 1}^{2N} T_{j_\gamma,m_\gamma}T_{j_\gamma,n_\gamma}^*
         \left\langle \boldsymbol{0} \left|c_{2p_{\ell}}..c_{2p_1}\left(\prod_{\gamma = 1}^{k} c_{m_{\gamma}}c_{n_{\gamma}}\right)c_{2p_1}..c_{2p_{\ell}} \right|\boldsymbol{0}\right\rangle.

    where :math:`c` are the majoranas fermions, :math:`y^*` is the target output state, :math:`x` is the system state
    and :math:`T` is the given transition matrix of the circuit or of the system evolution.

    .. raw:: latex

        \begin{tabular}{|cc|c|c|c|}
            \hline
            \multicolumn{2}{|c|}{\multirow{2}{*}{$k<j$}} & \multicolumn{3}{|c|}{$j$} \\
            \cline{3-5}
            & & $c_{m_{\beta}}$ & $c_{n_{\beta}}$ & $c_{2p_{\beta}}$ \\
            \hline
            \multirow{4}{*}{$k$} & $c_{m_{\alpha}}$ & $\qty(TBT^\mathsf{T})_{j_{\alpha},j_{\beta}}$ & $\qty(TBT^\dagger)_{j_{\alpha},j_{\beta}}$ & $\qty(TB)_{j_{\alpha},2p_{\beta}}$ \\
            & $c_{n_{\alpha}}$ & $\qty(T^*BT^\mathsf{T})_{j_{\alpha},j_{\beta}}$ & $\qty(T^*BT^\dagger)_{j_{\alpha},j_{\beta}}$ & $\qty(T^*B)_{j_{\alpha},2p_{\beta}}$ \\
            & $c_{2p_{\alpha}}$ & $\qty(BT^\mathsf{T})_{2p_{\alpha},j_{\beta}}$ & $\qty(BT^\dagger)_{2p_{\beta},j_{\beta}}$ & $\delta_{\alpha,\beta}$ \\
            \hline
        \end{tabular}

    # TODO: Make it a torch function to output the analytical grad instead of creating a huge computational graph
    # TODO:     during the current computation.
    # TODO: Tips for optimization: Maybe there is a way to use the sparsity of the block diagonal matrix to reduce
    # TODO:     the number of operations in the lookup table.

    :ivar ALL_2D_INDEXES: Stores all possible 2D indexes for a system with three rows.
    :type ALL_2D_INDEXES: np.ndarray
    :ivar ALL_1D_INDEXES: Contains converted 1D indexes corresponding to the 2D indexes.
    :type ALL_1D_INDEXES: np.ndarray
    :ivar p_bar: Progress bar object used for tracking computation, can be None if disabled.
    :type p_bar: Optional[Any]
    :ivar show_progress: Indicates if progress tracking through a progress bar should be enabled.
    :type show_progress: bool
    """

    ALL_2D_INDEXES = np.asarray(list(itertools.product(range(3), repeat=2)))
    ALL_1D_INDEXES = utils.math.convert_2d_to_1d_indexes(ALL_2D_INDEXES, n_rows=3)

    def __init__(
        self,
        transition_matrix: TensorLike,
        **kwargs,
    ):
        """
        Represents a class that contract the action of the given transition matrix on a free fermionic
        system into a tensor.

        :param transition_matrix: A tensor or array-like structure representing
            the transition matrix for the system.
        :param kwargs: Additional keyword arguments.
            Optional argument:
            - ``show_progress`` (bool): Indicates whether to show progress during
              operations. Default is ``False``.
        """
        self._transition_matrix = transition_matrix
        self.p_bar: Optional[tqdm.tqdm] = None
        self.show_progress: bool = kwargs.get("show_progress", False)

    def __call__(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        r"""
        Get the contracted tensor corresponding to target_binary_state and the system_state.

        :param system_state: State of the system
        :type system_state: Union[int, np.ndarray, sparse.sparray]
        :param target_binary_states: Target state of the system
        :type target_binary_states: Optional[np.ndarray]
        :param indexes_of_target_states: Indexes of the target state of the system
        :type indexes_of_target_states: Optional[np.ndarray]
        :return: The tensor of shape (2(h + k), 2(h + k)) where h is the hamming weight of the system state
            representing the action of the transition matrix on the system.
        :rtype: np.ndarray
        """
        self.show_progress = kwargs.get("show_progress", self.show_progress)
        return self.compute_observables_of_target_states(
            system_state, target_binary_states, indexes_of_target_states, **kwargs
        )

    def __getitem__(self, item: Union[Tuple[int, int], int]):
        """
        Retrieve the value at the specified index in a 2D table-like structure. This index
        can be provided as either a single integer or a tuple of two integers. In the case
        of a single integer, it is converted into 2D indices using a helper function.

        If `item` is a tuple, the first and second elements represent the row and column
        indices respectively. The retrieved value is determined by invoking the callable
        stored in a lookup table at the specified indices.

        :param item: The index to access the table's value. It can either be a tuple of
            two integers representing (row, column) indices or a single integer
            which will be converted into 2D indices.
        :return: The value retrieved by invoking the callable located at the specified
            index in the getter table.
        """
        if isinstance(item, int):
            item = utils.math.convert_1d_to_2d_indexes([item], n_rows=3)[0]
        i, j = item
        getter = self.getter_table[i][j]
        return getter()

    def compute_observables_of_target_states(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        """
        Computes observables for target quantum states based on the provided system state and configuration.

        This function processes the system state and a set of target binary states or their respective
        indexes to calculate observables. It constructs the necessary indices, computes intermediary terms
        required for observables, and combines them into the final observable tensor. The resulting
        observables capture quantum correlations and measurements over the target states.

        :param system_state: The state of the system, which can be an integer, a numpy array, or a sparse array.
        :param target_binary_states: Optional numpy array of the target binary states. Represents the
            configurations of the target quantum states.
        :param indexes_of_target_states: Optional numpy array of indices corresponding to target states. If
            provided, these indices specify which states to compute observables for.
        :param kwargs: Additional keyword arguments to configure or modify the behavior of the computations.
        :return: The computed tensor-like object representing the observables for the target states.
            The tensor reflects quantum measurements and correlations for the given configurations.
        """
        system_state, target_binary_states, indexes_of_target_states, initial_ndim = self._setup_inputs(
            system_state, target_binary_states, indexes_of_target_states, **kwargs
        )
        target_batch_size = target_binary_states.shape[0]
        bra_majorana_indexes, ket_majorana_indexes = self._get_bra_ket_indexes(system_state, target_binary_states)
        lt_indexes = self._get_lt_indexes(target_binary_states, ket_majorana_indexes)
        majorana_indexes = self._get_majorana_indexes(
            indexes_of_target_states, bra_majorana_indexes, ket_majorana_indexes
        )

        obs_size = majorana_indexes.shape[-1]
        obs_indices = np.stack(np.triu_indices(obs_size, k=1))
        lt_item_rows, lt_item_cols = (
            majorana_indexes[..., obs_indices[0]],
            majorana_indexes[..., obs_indices[1]],
        )

        # compute items needed for the observable
        all_lt_indexes = np.stack((lt_indexes[..., obs_indices[0]], lt_indexes[..., obs_indices[1]]), axis=-1)
        all_lt_indexes_raveled = all_lt_indexes.reshape(-1, 2)

        all_lt_indexes_raveled_1d = utils.math.convert_2d_to_1d_indexes(all_lt_indexes_raveled, n_rows=3)
        all_lt_indexes_1d = all_lt_indexes_raveled_1d.reshape(*all_lt_indexes.shape[:-1])
        unique_lt_indexes_raveled_1d = np.fromiter(set(all_lt_indexes_raveled_1d), dtype=int)
        unique_lt_indexes_raveled_2d = utils.math.convert_1d_to_2d_indexes(unique_lt_indexes_raveled_1d, n_rows=3)

        lt_items = self._compute_stack_and_pad_items(unique_lt_indexes_raveled_2d, close_p_bar=False)
        new_all_lt_indexes = self._extend_unique_indexes_to_all_indexes(all_lt_indexes_1d, unique_lt_indexes_raveled_1d)
        lt_items = lt_items.reshape(lt_items.shape[0], -1, *lt_items.shape[-2:])

        batch_size = [self.batch_size] if self.batch_size else [1]
        obs_shape = [target_batch_size] + batch_size + [obs_size, obs_size]
        obs = qml.math.convert_like(np.zeros(obs_shape, dtype=complex), self.transition_matrix)
        obs[..., obs_indices[0], obs_indices[1]] = qml.math.transpose(
            lt_items[new_all_lt_indexes, ..., lt_item_rows, lt_item_cols], (0, -1, -2)
        )

        self.p_bar_set_postfix_str("Finishing the computation of the observable.")
        obs = obs - qml.math.einsum("...ij->...ji", obs)
        if self.batch_size is None or self.batch_size == 0:
            obs = obs[:, 0, ...]
        if initial_ndim == 1:
            obs = obs[0, ...]
        self.p_bar_set_postfix_str("Finished the computation of the observable.")
        self.close_p_bar()
        return obs

    def compute_observable_of_target_state(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_state: Optional[np.ndarray] = None,
        indexes_of_target_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TensorLike:
        """
        Compute the observable of a target state from the given system state.

        This method calculates the quantum observable corresponding to a specific target
        state defined by the binary representation or the specified indices. If the
        resulting observable has multiple dimensions beyond the expected output shape,
        it will be reshaped accordingly. This accounts for scenarios where batch
        processing is not specified or when multi-dimensional shapes arise.

        :param system_state: The current state of the system, which can be an integer,
            a Numpy array, or a sparse array.
        :param target_binary_state: Optional Numpy array representing the binary
            configuration of the target state.
        :param indexes_of_target_state: Optional Numpy array containing the indices
            corresponding to the target state.
        :param kwargs: Arbitrary keyword arguments for additional configurations or
            parameters.
        :return: A tensor-like object representing the observable(s) of the target
            state. The shape of the output may vary depending on the dimensional nature
            of the input and batch configurations.
        """
        observables = self.compute_observables_of_target_states(
            system_state=system_state,
            target_binary_states=target_binary_state,
            indexes_of_target_states=indexes_of_target_state,
            **kwargs,
        )
        if len(qml.math.shape(observables)) > 2 and not self.batch_size:
            observables = qml.math.reshape(observables, (-1, *qml.math.shape(observables)[-2:]))[0]
        elif len(qml.math.shape(observables)) > 3:
            observables = qml.math.reshape(observables, (-1, *qml.math.shape(observables)[-3:]))[0]
        return observables

    def update_p_bar(self, *args, **kwargs):
        """
        Updates the progress bar with the given arguments and refreshes its state. This method
        safeguards the operation if the progress bar is not initialized.

        :param args: Positional arguments to pass to the progress bar's update method.
        :param kwargs: Keyword arguments to pass to the progress bar's update method.
        :return: None
        """
        if self.p_bar is None:
            return
        self.p_bar.update(*args, **kwargs)
        self.p_bar.refresh()

    def p_bar_set_n(self, n: int):
        if self.p_bar is not None:
            self.p_bar.n = n
            self.p_bar.refresh()

    def initialize_p_bar(self, *args, **kwargs):
        kwargs.setdefault("disable", not self.show_progress)
        if self.p_bar is None and not self.show_progress:
            return None
        self.p_bar = tqdm.tqdm(*args, **kwargs)
        return self.p_bar

    def p_bar_set_postfix(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix(*args, **kwargs)
            self.p_bar.refresh()

    def p_bar_set_postfix_str(self, *args, **kwargs):
        if self.p_bar is not None:
            self.p_bar.set_postfix_str(*args, **kwargs)
            self.p_bar.refresh()

    def close_p_bar(self):
        if self.p_bar is not None:
            self.p_bar.close()

    def get_c_d_alpha__c_d_beta(self):
        return self.c_d_alpha__c_d_beta

    def get_c_d_alpha__c_e_beta(self):
        return self.c_d_alpha__c_e_beta

    def get_c_d_alpha__c_2p_beta_m1(self):
        return self.c_d_alpha__c_2p_beta_m1

    def get_c_e_alpha__c_d_beta(self):
        return self.c_e_alpha__c_d_beta

    def get_c_e_alpha__c_e_beta(self):
        return self.c_e_alpha__c_e_beta

    def get_c_e_alpha__c_2p_beta_m1(self):
        return self.c_e_alpha__c_2p_beta_m1

    def get_c_2p_alpha_m1__c_d_beta(self):
        return self.c_2p_alpha_m1__c_d_beta

    def get_c_2p_alpha_m1__c_e_beta(self):
        return self.c_2p_alpha_m1__c_e_beta

    def get_c_2p_alpha_m1__c_2p_beta_m1(self):
        return self.c_2p_alpha_m1__c_2p_beta_m1

    def _setup_inputs(
        self,
        system_state: Union[int, np.ndarray, sparse.sparray],
        target_binary_states: Optional[np.ndarray] = None,
        indexes_of_target_states: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        r"""
        Setup the inputs for the compute_observables_of_target_states method.
        This includes reshaping the inputs and checking their validity.

        :param system_state: The state of the system.
        :type system_state: Union[int, np.ndarray, sparse.sparray]
        :param target_binary_states: The target states to compute the probability from.
        :type target_binary_states: Optional[np.ndarray]
        :param indexes_of_target_states: The index of the particles of the target states.
        :type indexes_of_target_states: Optional[np.ndarray]
        :param kwargs: Additional keywords arguments

        :return: The processed inputs: (system_state, target_binary_states, indexes_of_target_states, initial_ndim)
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        """
        initial_ndim = None
        if target_binary_states is not None:
            target_binary_states = np.asarray(target_binary_states)
            initial_ndim = len(qml.math.shape(target_binary_states))
            target_binary_states = target_binary_states.reshape(-1, target_binary_states.shape[-1])

        if indexes_of_target_states is not None:
            indexes_of_target_states = np.asarray(indexes_of_target_states)
            initial_ndim = len(qml.math.shape(indexes_of_target_states))
            indexes_of_target_states = indexes_of_target_states.reshape(-1, indexes_of_target_states.shape[-1])

        if target_binary_states is None and indexes_of_target_states is None:
            target_binary_states = np.array(
                [
                    [
                        1,
                    ]
                ]
            )
            indexes_of_target_states = np.array(
                [
                    [
                        0,
                    ]
                ]
            )
        elif target_binary_states is not None and indexes_of_target_states is None:
            indexes_of_target_states = np.arange(target_binary_states.shape[-1], dtype=int)[np.newaxis, :].repeat(
                target_binary_states.shape[0], axis=0
            )
        elif target_binary_states is None and indexes_of_target_states is not None:
            target_binary_states = np.ones(indexes_of_target_states.shape[-1], dtype=int)[np.newaxis, :].repeat(
                indexes_of_target_states.shape[0], axis=0
            )

        self._assert_binary(system_state)
        self._assert_binary(target_binary_states)
        return (
            system_state,
            target_binary_states,
            indexes_of_target_states,
            initial_ndim,
        )

    def _get_bra_ket_indexes(self, system_state: np.ndarray, target_binary_states: np.ndarray, **kwargs):
        """
        Determines the bra and ket Majorana indexes for a given quantum system state
        and specified binary target states. This function operates on predefined
        quantum binary states and computes their respective Majorana representations
        to enable further quantum state manipulations.

        :param system_state:
            The current quantum system state represented as a numpy ndarray.
        :param target_binary_states:
            A numpy ndarray representing the target quantum binary states, where
            each binary state in the array corresponds to a specific system state
            configuration.
        :param kwargs:
            Optional keyword arguments for additional parameterization.

        :return:
            A tuple containing:

            - `bra_majorana_indexes` (numpy ndarray): The Majorana indexes
              corresponding to the bra (conjugate transpose) quantum state, repeated
              for each target binary state.
            - `ket_majorana_indexes` (numpy ndarray): The Majorana indexes
              corresponding to the given ket (initial) state, repeated for each
              target binary state.
        """
        target_batch_size = target_binary_states.shape[0]
        ket_majorana_indexes = utils.decompose_binary_state_into_majorana_indexes(system_state)
        bra_majorana_indexes = list(reversed(ket_majorana_indexes))

        ket_majorana_indexes = np.asarray(ket_majorana_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        bra_majorana_indexes = np.asarray(bra_majorana_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        return bra_majorana_indexes, ket_majorana_indexes

    def _get_lt_indexes(
        self,
        target_binary_states: np.ndarray,
        ket_majorana_indexes: np.ndarray,
    ):
        """
        This function generates index arrays for a specific operation by combining input binary
        states and Majorana index information. It creates unmeasured class indexes
        and measured class indexes by using the provided target binary states and Majorana indexes,
        then concatenates them to produce the final array.

        :param target_binary_states: A 2D numpy array where each row represents a binary state configuration.
        :param ket_majorana_indexes: A 2D numpy array where each row depicts Majorana index values
            corresponding to the quantum Majorana fermion system.
        :return: A 2D numpy array of integers representing the concatenated indexes,
            consisting of unmeasured class indexes, measured class indexes derived from target binary
            states, and unmeasured class indexes.
        """
        target_batch_size = target_binary_states.shape[0]
        unmeasured_cls_indexes = [2 for _ in range(ket_majorana_indexes.shape[-1])]
        unmeasured_cls_indexes = np.asarray(unmeasured_cls_indexes)[np.newaxis, :].repeat(target_batch_size, axis=0)
        measure_cls_indexes = np.asarray([np.array([[b, 1 - b] for b in t]).flatten() for t in target_binary_states])
        lt_indexes = np.concatenate(
            [unmeasured_cls_indexes, measure_cls_indexes, unmeasured_cls_indexes],
            axis=-1,
        )
        return lt_indexes.astype(int)

    def _get_majorana_indexes(
        self,
        indexes_of_target_states: np.ndarray,
        bra_majorana_indexes: np.ndarray,
        ket_majorana_indexes: np.ndarray,
    ):
        """
        This function computes the Majorana indexes based on the given target states,
        bra Majorana indexes, and ket Majorana indexes.

        The input arrays are combined into a single array of Majorana indexes by
        concatenating the bra Majorana indexes, a measurement component derived from
        the target states, and the ket Majorana indexes, along the last axis. The
        resulting array is returned as an integer type for further computation.

        :param indexes_of_target_states: np.ndarray
            Array containing the indexes of target states. Each target state is
            represented as an array of integers, which forms part of the measurement
            component.
        :param bra_majorana_indexes: np.ndarray
            Array containing the bra Majorana indexes, which are included as the
            initial part of the resulting concatenation.
        :param ket_majorana_indexes: np.ndarray
            Array containing the ket Majorana indexes, which are appended as part of
            the final concatenation.
        :return: np.ndarray
            A single concatenated array of Majorana indexes. The array is of
            integer type and contains the combined information of bra Majorana
            indexes, measurement component, and ket Majorana indexes.
        """
        measure_indexes = np.asarray([np.array([[i, i] for i in t]).flatten() for t in indexes_of_target_states])
        majorana_indexes = np.concatenate(
            [bra_majorana_indexes, measure_indexes, ket_majorana_indexes], axis=-1
        ).astype(int)
        return majorana_indexes

    def _compute_items(self, indexes: Sequence[Tuple[int, int]], close_p_bar: bool = True) -> List[TensorLike]:
        """
        Compute the items of the lookup table corresponding to the indexes.

        :param indexes: Indexes of the items to compute.
        :param close_p_bar: Whether to close the progress bar.
        :return: The items of the lookup table corresponding to the indexes.
        """
        self.initialize_p_bar(total=len(indexes), initial=0, desc="Computing Lookup Table Items")
        items = []
        indexes = np.asarray(indexes)
        indexes = indexes.reshape(-1, 2)
        for i, j in indexes:
            items.append(self[i, j])
            self.update_p_bar()
        if close_p_bar:
            self.close_p_bar()  # pragma: no cover
        return items

    def _compute_stack_and_pad_items(
        self,
        indexes: Iterable[Tuple[int, int]],
        pad_value: numbers.Number = 0.0,
        close_p_bar: bool = True,
    ) -> TensorLike:
        """
        Computes a stacked tensor of items based on the given indexes and pads the tensors if
        necessary to create consistent dimensions. Padding is applied with the specified
        `pad_value` where needed. Additionally, a progress bar can be optionally closed.

        :param indexes: An iterable of tuples where each tuple represents an index combination
        :param pad_value: The value used to pad tensors to match the maximum dimension for
                          stacking. Default is 0.0.
        :type pad_value: numbers.Number
        :param close_p_bar: A boolean flag indicating whether the progress bar should be closed
                            after computation. Default is True.
        :type close_p_bar: bool
        :return: The stacked tensor of computed items, padded if necessary, conforming to
                 consistent dimensions.
        :rtype: TensorLike
        """
        items = self._compute_items(indexes, close_p_bar=close_p_bar)
        items_shapes = [qml.math.shape(i) for i in items]
        items_has_same_shape = all([i == items_shapes[0] for i in items_shapes])

        if items_has_same_shape:
            items = qml.math.stack(items)
        else:
            # need to pad the items to max dim in each dimension and stack them
            max_dim_0, max_dim_1 = max([i[-2] for i in items_shapes]), max([i[-1] for i in items_shapes])
            for i, (item, item_shape) in enumerate(zip(items, items_shapes)):
                new_shape = list(item_shape)
                new_shape[-2] = max_dim_0
                new_shape[-1] = max_dim_1
                new_item = qml.math.convert_like(np.full(new_shape, fill_value=pad_value, dtype=complex), item)
                new_item[..., : item_shape[-2], : item_shape[-1]] = item
                items[i] = new_item
            items = qml.math.stack(items)
        return items

    @staticmethod
    def _extend_unique_indexes_to_all_indexes(
        all_indexes,
        unique_indexes,
    ):
        """
        Transforms `all_indexes` array by mapping its values according to their positions
        in the `unique_indexes` array. Every value in `all_indexes` will be replaced by the
        index of its corresponding position in `unique_indexes`.

        :param all_indexes: numpy array representing the initial index values
        :param unique_indexes: numpy array containing unique index values to serve as mapping references
        :return: numpy array with values replaced based on positions in `unique_indexes`
        """
        new_all_indexes = np.empty(all_indexes.shape, dtype=int)
        for i, j in enumerate(unique_indexes):
            new_all_indexes = np.where(all_indexes == j, i, new_all_indexes)
        return new_all_indexes

    @staticmethod
    def _assert_binary(binary_state: np.ndarray) -> bool:
        r"""
        Check if the binary state contains only zeros or ones. If not, a value error will be raised.

        :param binary_state: Input binary state.
        :type binary_state: np.ndarray

        :return: Whether the input state is binary or not.
        :rtype: bool

        :raises: ValueError
        """
        unique_values = np.unique(binary_state)
        if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"The binary state must contain only zeros and ones. Currently contains: {unique_values}")
        return True

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @cached_property
    def n_particles(self):
        return qml.math.shape(self.transition_matrix)[-2]

    @cached_property
    def batch_size(self):
        if qml.math.ndim(self.transition_matrix) < 3:
            return 0
        return qml.math.shape(self.transition_matrix)[0]

    @cached_property
    def block_diagonal_matrix(self):
        r"""
        The block diasgonal matrix :math:`B` is a :math:`2N \times 2N` Hermitian matrix block diagonal matrix containing
        :math:`N` number of :math:`2 \times 2` blocks

        .. math::
            B = \left(
                \begin{matrix}
                    1 & i & & & &\\
                    -i & 1 & & & &\\
                     & & & \ddots & & \\
                     & & & & 1 & i
                \end{matrix}
            \right)
            = \bigoplus_{l = 1}^{N} \left(
            \begin{matrix}
                1 & i \\
                -i & 1
            \end{matrix}
            \right).

        :return: the :math:`B` matrix.
        """
        return utils.get_block_diagonal_matrix(self.n_particles)

    @cached_property
    def block_bm_transition_transpose_matrix(self):
        self.p_bar_set_postfix_str("Computing BT^T matrix.")
        return qml.math.einsum(f"ij,...kj->...ik", self.block_diagonal_matrix, self.transition_matrix)

    @cached_property
    def block_bm_transition_dagger_matrix(self):
        self.p_bar_set_postfix_str("Computing BT^dagger matrix.")
        return qml.math.einsum(
            f"ij,...kj->...ik",
            self.block_diagonal_matrix,
            qml.math.conjugate(self.transition_matrix),
        )

    @cached_property
    def transition_bm_block_matrix(self):
        self.p_bar_set_postfix_str("Computing TB matrix.")
        return qml.math.einsum(f"...ij,jk->...ik", self.transition_matrix, self.block_diagonal_matrix)

    @cached_property
    def shape(self) -> Tuple[int, int]:
        return 3, 3

    @cached_property
    def c_d_alpha__c_d_beta(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            self.transition_matrix,
        )

    @cached_property
    def c_d_alpha__c_e_beta(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_d_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pj,...kj->...pk",
            self.transition_bm_block_matrix,
            qml.math.conjugate(self.transition_matrix),
        )

    @property
    def c_d_alpha__c_2p_beta_m1(self) -> TensorLike:
        return self.transition_bm_block_matrix

    @cached_property
    def c_e_alpha__c_d_beta(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_d_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_transpose_matrix,
        )

    @cached_property
    def c_e_alpha__c_e_beta(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_e_beta.")
        return qml.math.einsum(
            f"...pi,...ik->...pk",
            qml.math.conjugate(self._transition_matrix),
            self.block_bm_transition_dagger_matrix,
        )

    @cached_property
    def c_e_alpha__c_2p_beta_m1(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_e_alpha__c_2p_beta_m1.")
        return qml.math.einsum(
            f"...pi,ij->...pj",
            qml.math.conjugate(self._transition_matrix),
            self.block_diagonal_matrix,
        )

    @property
    def c_2p_alpha_m1__c_d_beta(self) -> TensorLike:
        return self.block_bm_transition_transpose_matrix

    @property
    def c_2p_alpha_m1__c_e_beta(self) -> TensorLike:
        return self.block_bm_transition_dagger_matrix

    @cached_property
    def c_2p_alpha_m1__c_2p_beta_m1(self) -> TensorLike:
        self.p_bar_set_postfix_str("Computing c_2p_alpha_m1__c_2p_beta_m1.")
        if self.batch_size > 0:
            size = qml.math.shape(self.transition_matrix)[-1]
            shape = ([self.batch_size] if self.batch_size else []) + [size, size]
            matrix = pnp.zeros(shape, dtype=complex)
            matrix[..., :, :] = qml.math.eye(size, dtype=complex)
        else:
            matrix = np.eye(self.transition_matrix.shape[-1])
        matrix = qml.math.convert_like(matrix, self.transition_matrix)
        return matrix

    @cached_property
    def getter_table(self) -> List[List[callable]]:
        return [
            [
                self.get_c_d_alpha__c_d_beta,
                self.get_c_d_alpha__c_e_beta,
                self.get_c_d_alpha__c_2p_beta_m1,
            ],
            [
                self.get_c_e_alpha__c_d_beta,
                self.get_c_e_alpha__c_e_beta,
                self.get_c_e_alpha__c_2p_beta_m1,
            ],
            [
                self.get_c_2p_alpha_m1__c_d_beta,
                self.get_c_2p_alpha_m1__c_e_beta,
                self.get_c_2p_alpha_m1__c_2p_beta_m1,
            ],
        ]

    @cached_property
    def stacked_items(self):
        return self._compute_stack_and_pad_items(self.ALL_2D_INDEXES, close_p_bar=False)
