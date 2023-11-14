from typing import Type, List
import os

import numpy as np
import pennylane as qml
from tests.test_nif_device import devices_init, single_matchgate_circuit
from msim import mps, utils
import matplotlib.pyplot as plt
import tqdm


def single_measure_p1(params):
    nif_device, qubit_device = devices_init()

    nif_qnode = qml.QNode(single_matchgate_circuit, nif_device)
    qubit_qnode = qml.QNode(single_matchgate_circuit, qubit_device)

    nif_probs = nif_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy())
    qubit_probs = qubit_qnode(mps.MatchgatePolarParams.parse_from_any(params).to_numpy())
    return nif_probs[-1], qubit_probs[-1]


def compute_absolute_errors(n_points: int = 10_000, seed: int = 0, **kwargs):
    params_type = kwargs.get('params_type', mps.MatchgatePolarParams)
    save_path = kwargs.get("save_path", None)
    if save_path is not None and os.path.exists(save_path):
        data = np.load(save_path)
        params_list = [params_type.parse_from_any(params) for params in data['params_list']]
        return data['errors'], params_list

    np.random.seed(seed)
    errors, params_list = [], []
    for _ in tqdm.trange(n_points):
        # h_params = mps.MatchgateHamiltonianCoefficientsParams.random()
        # h_params.epsilon = 0.0
        h_params = mps.MatchgatePolarParams(
            r0=0,
            r1=np.random.rand(),
            theta0=np.random.rand() * 2 * np.pi,
            theta1=np.random.rand() * 2 * np.pi,
            theta2=np.random.rand() * 2 * np.pi,
            theta3=np.random.rand() * 2 * np.pi,
        )
        params = mps.transfer_functions.params_to(h_params, params_type)
        nif_p1, qubit_p1 = single_measure_p1(params)
        errors.append(np.abs(nif_p1 - qubit_p1))
        params_list.append(params)
    errors = np.asarray(errors)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, errors=errors, params_list=[params.to_numpy() for params in params_list])

    return errors, params_list


def show_absolute_error_distribution(**kwargs):
    errors, params_list = kwargs.get('errors', None), kwargs.get('params_list', None)
    if errors is None or params_list is None:
        errors, params_list = compute_absolute_errors(n_points=kwargs.get('n_points', 10_000))
    
    mean, std = np.mean(errors), np.std(errors)
    mean_std_str = rf"{mean:.3f} $\pm$ {std:.3f}"
    
    fig, ax = plt.subplots()
    # Histogram:
    # Bin it
    n, bin_edges = np.histogram(errors, 100)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n / float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:] + bin_edges[:-1]) / 2.
    # Compute the bin-width
    bin_width = bin_edges[1] - bin_edges[0]
    # Plot the histogram as a bar plot
    ax.bar(bin_middles, bin_probability, width=bin_width)

    ax.set_xlabel("Absolute error [-]")
    ax.set_ylabel("Probability [-]")
    ax.set_title(f"Absolute error distribution ({mean_std_str})")

    if kwargs.get('tight_layout', True):
        plt.tight_layout()
    if kwargs.get('save', True):
        save_path = kwargs.get(
            'save_path',
            os.path.join(os.path.dirname(__file__), "figures", 'absolute_error_distribution.png')
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if kwargs.get('show', True):
        plt.show()
    return fig, ax


def show_correlation_between_error_and_params(
        params_types: List[Type[mps.MatchgateParams]] = None,
        **kwargs
):
    if params_types is None:
        params_types = mps.MatchgateParams.__subclasses__()
    elif not isinstance(params_types, list):
        params_types = [params_types]

    errors, params_list = kwargs.get('errors', None), kwargs.get('params_list', None)
    if errors is None or params_list is None:
        errors, params_list = compute_absolute_errors(n_points=kwargs.get('n_points', 10_000))

    # make square grid of subplots
    n_plots = len(params_types)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=False, sharey=True)
    axes = np.ravel(np.asarray([axes]))
    for ax, params_type in zip(axes, params_types):
        __params_list = np.asarray([params_type.parse_from_any(params).to_numpy() for params in params_list])
        correlations = [
            np.corrcoef(__params_list[:, i], errors)[0, 1]
            for i in range(params_type.N_PARAMS)
        ]
        ax.bar(np.arange(params_type.N_PARAMS), correlations)
        params_names = [rf"${s.name}$" for s in params_type.to_sympy()]
        ax.set_xticks(np.arange(params_type.N_PARAMS))
        ax.set_xticklabels(params_names)
        ax.set_xlabel("Parameters [-]")
        ax.set_ylabel("Correlation coefficient [-]")
        ax.set_title(f"{utils.camel_case_to_spaced_camel_case(params_type.get_short_name())}")

    # disable unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    if kwargs.get('tight_layout', True):
        plt.tight_layout()
    if kwargs.get('save', True):
        save_path = kwargs.get(
            'save_path',
            os.path.join(os.path.dirname(__file__), "figures", 'correlation_between_error_and_params.png')
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if kwargs.get('show', True):
        plt.show()
    return fig, axes


if __name__ == '__main__':
    e, p = compute_absolute_errors(
        n_points=1_000,
        # save_path=os.path.join(os.path.dirname(__file__), "cache", 'abs_errors.npz')
    )
    show_absolute_error_distribution(errors=e, params_list=p, show=True, save=True)
    show_correlation_between_error_and_params(errors=e, params_list=p, show=True, save=True)
