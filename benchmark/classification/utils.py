import functools
import os
import warnings
from collections import defaultdict
from functools import partial
from typing import Sequence, Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score

MPL_RC_DEFAULT_PARAMS = {
    "font.size": 18,
    "legend.fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.major.width": 3,
    'xtick.minor.visible': True,
    'xtick.minor.size': 6.0,
    'xtick.minor.width': 2.0,
    "ytick.direction": "in",
    "ytick.major.width": 3,
    "ytick.major.size": 12,
    'ytick.minor.visible': True,
    'ytick.minor.size': 6.0,
    'ytick.minor.width': 2.0,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "DejaVu Sans",
        "Arial",
        "Tahoma",
        "calibri",
    ],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "errorbar.capsize": 2,
}

MPL_RC_BIG_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 36,
    "legend.fontsize": 22,
    "lines.markersize": 12,
}}

MPL_RC_SMALL_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 12,
    "legend.fontsize": 10,
}}
mStyles = [
    "o", "v", "*", "X", "^", "<", ">", "+", ",", "1", "2", "3", "4", "8", "s", "p", "P", "h", "H", ".", "x", "D", "d",
    "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
]

BIG_O_STR = r"\mathcal{O}"

def constant(x, c):
    return np.full_like(x, c)


def linear(x, m, c):
    return m * x + c


def polynomial(x, *coefficients):
    return np.sum([coefficients[i] * x**i for i in range(len(coefficients))], axis=0)


def exponential(x, a, b, c):
    return c + a * np.exp(b * x)


def exponential2(x, a, b, c):
    return c + a * 2**(b * x)


def find_complexity(x, y, **kwargs):
    """
    Find if y scales with x as a constant, linear, polynomial, or exponential function.

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    x_lbl = kwargs.get('x_lbl', 'x')
    functions = {
        'constant': constant,
        'linear': linear,
        'polynomial': polynomial,
        'exponential': exponential,
        'exponential2': exponential2,
    }
    p0_dict = {
        'polynomial': np.random.rand(min(x[-1], len(x))),
    }
    # Fit data to each type of function
    popt_dict = {
        f_name: curve_fit(f, x, y, p0_dict.get(f_name, None))[0]
        for f_name, f in functions.items()
    }
    labels = {
        'constant': f"~$O({popt_dict['constant'][0]:.4f})$",
        'linear': f"~$O({popt_dict['linear'][0]:.4f}{x_lbl})$",
        'polynomial': f"~$O({popt_dict['polynomial'][-1]:.4f}{x_lbl}^{len(popt_dict['polynomial'])-1})$",
        'exponential': f"~$O({popt_dict['exponential'][0]:.4f}e^{{ {popt_dict['exponential'][1]:.4f}{x_lbl} }})$",
        'exponential2': f"~$O({popt_dict['exponential2'][0]*2:.4f}^{{ {popt_dict['exponential2'][1]:.4f}{x_lbl} }})$",
    }

    # Predict y values using fitted functions
    y_fit_dict = {
        f_name: functions[f_name](x, *popt)
        for f_name, popt in popt_dict.items()
    }
    # Determine which function fits best based on R-squared value
    r_squared_values = {
        f_name: r2_score(y, y_fit)
        for f_name, y_fit in y_fit_dict.items()
    }
    best_fit = max(r_squared_values, key=r_squared_values.get)
    return {
        'best_fit': best_fit,
        'popt': popt_dict[best_fit],
        'r_squared': r_squared_values[best_fit],
        'func': functions[best_fit],
        'y_fit': y_fit_dict[best_fit],
        'label': labels[best_fit],
    }


def cexp(x, a, b, c):
    return c + a ** (b * x)


def aexp(x, a, b):
    return b + a ** x


def ten_exp(x, a, b, c):
    return b + 10 ** (a * x + c)


def exp_fit(x, y, **kwargs):
    """
    Fit data to an exponential function of the form y = c + a * 2^(b * x)

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    base = kwargs.pop('base', 2)
    func = lambda _x, a, c: c * base ** (a * _x)
    x_lbl = kwargs.pop('x_lbl', 'x')
    # p0 = [1.0, y[0], 0.0]
    popt, pcov = curve_fit(func, x, y)
    r2 = r2_score(y, func(x, *popt))
    # label = f"~$O({popt[0]*2:.2f}^{{ {popt[1]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    label = f"~${BIG_O_STR}({base}^{{ {popt[0]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    # label = f"~$O({popt[1]:.2f}{x_lbl}^{{ {popt[0]:.2f} }})$: $R^2={r2:.2f}$"
    # label = f"~$O({popt[0]*2:.2f}^{{ {x_lbl} }})$: $R^2={r2:.2f}$"
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        "r2": r2,
        'label': label,
        "func": func,
    }


def poly(x, a, b, c):
    return c + b * (x ** a)


def poly_fit(x, y, **kwargs):
    """
    Fit data to an exponential function of the form y = c + a * 2^(b * x)

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    if kwargs.get("add_bias", False):
        func = poly
    else:
        func = lambda _x, a, b: b * (_x ** a)
    x_lbl = kwargs.pop('x_lbl', 'x')
    # p0 = [1.0, 1.0, y[0]]
    popt, pcov = curve_fit(func, x, y)
    r2 = r2_score(y, func(x, *popt))
    # label = f"~$O({popt[0]*2:.2f}^{{ {popt[1]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    # label = f"~$O(10^{{ {popt[0]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    label = f"~${BIG_O_STR}({popt[1]:.2f}{x_lbl}^{{ {popt[0]:.2f} }})$: $R^2={r2:.2f}$"
    # label = f"~$O({popt[0]*2:.2f}^{{ {x_lbl} }})$: $R^2={r2:.2f}$"
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        "r2": r2,
        'label': label,
        "func": func,
    }


def lin_fit(x, y, **kwargs):
    func = lambda _x, m, b: m * _x + b
    x_lbl = kwargs.pop('x_lbl', 'x')
    popt, pcov = curve_fit(func, x, y)
    r2 = r2_score(y, func(x, *popt))
    # label = f"~$O({popt[0]*2:.2f}^{{ {popt[1]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    # label = f"~$O(10^{{ {popt[0]:.2f}{x_lbl} }})$: $R^2={r2:.2f}$"
    # label = f"~${BIG_O_STR}({popt[1]:.2f}{x_lbl}^{{ {popt[0]:.2f} }})$: $R^2={r2:.2f}$"
    label = f"~${BIG_O_STR}({popt[0]:.2f}{x_lbl})$: $R^2={r2:.2f}$"
    # label = f"~$O({popt[0]*2:.2f}^{{ {x_lbl} }})$: $R^2={r2:.2f}$"
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        "r2": r2,
        'label': label,
        "func": func,
    }

def loglin_fit(x, y, **kwargs):
    """
    Fit data to an

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    base = kwargs.pop('base', 2)
    np_log_func = lambda _x: np.log(_x) / np.log(base)
    func = linear
    log_func = lambda _x, m, c: np_log_func(m * _x + c)
    x_lbl = kwargs.pop('x_lbl', 'x')
    popt, pcov = curve_fit(func, x, np_log_func(y), **kwargs)
    r2 = r2_score(y, func(x, *popt))
    label = f"~$O({popt[0]:.2f}{x_lbl})$: $R^2={r2:.2f}$"
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        "r2": r2,
        'label': label,
        "func": log_func,
    }


def log_fit(x, y, **kwargs):
    """
    Fit data to an exponential function of the form y = c + a * 2^(b * x)

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    base = kwargs.pop('base', 2)
    if kwargs.get("add_bias", True):
        func = lambda _x, a, c: c + a * (np.log(_x) / np.log(base))
    else:
        func = lambda _x, a: a * (np.log(_x) / np.log(base))
    x_lbl = kwargs.pop('x_lbl', 'x')
    popt, pcov = curve_fit(func, x, y)
    r2 = r2_score(y, func(x, *popt))
    label = f"~${BIG_O_STR}({popt[0]:.2f}log_{{{base}}}({x_lbl}))$: $R^2={r2:.2f}$"
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        "r2": r2,
        'label': label,
        "func": func,
    }


def load_mnist1d():
    import requests, pickle

    url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
    r = requests.get(url, allow_redirects=True)
    open('./mnist1d_data.pkl', 'wb').write(r.content)

    with open('./mnist1d_data.pkl', 'rb') as handle:
        data = pickle.load(handle)
    data["data"] = np.concatenate([data["x"], data["x_test"]], axis=0)
    data["target"] = np.concatenate([data["y"], data["y_test"]], axis=0)
    return data


def number_axes(axes: Sequence[plt.Axes], **kwargs) -> Sequence[plt.Axes]:
    """
    Number the axes.

    :param axes: Axes to number.
    :type axes: Sequence[plt.Axes]
    :param kwargs: Keyword arguments.

    :keyword str num_type: Type of number to display. Can be either "alpha" or "numeric".
    :keyword int start: Number to start with.
    :keyword float x: x position of the number in the axes coordinate (see ax.transAxes). Default is 0.0.
    :keyword float y: y position of the number in the axes coordinate (see ax.transAxes). Default is 1.2.
    :keyword float fontsize: Font size of the number. Default is 12.
    :keyword str fontweight: Font weight of the number. Default is "bold".
    :keyword method: Method to use to number the axes. Available methods are "text", "title" and "set_title".
        The "text" method will add a text to the axes. The "title" method will add the number to the existing title.
        The "set_title" method will set the title of the axes, so the existing title will be overwritten.
        Default is "text".

    :return: The axes with the number.
    :rtype: Sequence[plt.Axes]
    """
    axes_view = np.ravel(np.asarray(axes))
    num_type = kwargs.get("num_type", "alpha").lower()
    mth = kwargs.get("method", "text").lower()
    start = kwargs.get("start", 0)
    if num_type == "alpha":
        axes_numbers = [chr(i) for i in range(97 + start, 97 + len(axes_view) + start)]
    elif num_type == "numeric":
        axes_numbers = [str(i) for i in range(1 + start, len(axes_view) + 1 + start)]
    else:
        raise ValueError(f"Unknown num_type {num_type}.")
    for i, ax in enumerate(axes_view):
        if mth == "text":
            ax.text(
                kwargs.get("x", 0.0), kwargs.get("y", 1.2),
                f"({axes_numbers[i]})",
                transform=ax.transAxes, fontsize=kwargs.get("fontsize", 12),
                fontweight=kwargs.get("fontweight", 'bold'), va='top'
            )
        elif mth == "title":
            ax.set_title(
                f"({axes_numbers[i]}) {ax.get_title()}",
                fontsize=kwargs.get("fontsize", 12),
                fontweight=kwargs.get("fontweight", 'bold'),
                loc=kwargs.get("loc", "left"),
            )
        elif mth == "set_title":
            ax.set_title("")
            ax.set_title(
                f"({axes_numbers[i]})",
                fontsize=kwargs.get("fontsize", 12),
                fontweight=kwargs.get("fontweight", 'bold'),
                loc=kwargs.get("loc", "left"),
            )
    return axes


def save_on_exit(_method=None, *, save_func_name="to_pickle", save_args=(), **save_kwargs):
    """
    Decorator for a method that saves the object on exit.

    :param _method: The method to decorate.
    :type _method: Callable
    :param save_func_name: The name of the method that saves the object.
    :type save_func_name: str
    :param save_args: The arguments of the save method.
    :type save_args: tuple
    :param save_kwargs: The keyword arguments of the save method.
    :return: The decorated method.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Failed to run method {method.__name__}: {e}", RuntimeWarning)
                raise e
            finally:
                self = args[0]
                if not hasattr(self, save_func_name):
                    warnings.warn(
                        f"The object {self.__class__.__name__} does not have a save method named {save_func_name}.",
                        RuntimeWarning
                    )
                try:
                    getattr(self, save_func_name)(*save_args, **save_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to save object {self.__class__.__name__}: {e} with {save_func_name} method",
                        RuntimeWarning
                    )
                    # raise e

        # wrapper.__name__ = method.__name__ + "@save_on_exit"
        wrapper.__name__ = method.__name__
        return wrapper

    if _method is None:
        return decorator
    else:
        return decorator(_method)


def get_gram_predictor(cls, kernel, x_train, **kwargs):
    def predictor(x_test):
        return cls.predict(kernel.pairwise_distances(x_test, x_train, **kwargs))
    return predictor


class KPredictorContainer:
    def __init__(self, name: str = ""):
        self.name = name
        self.container = defaultdict(dict)

    def get(self, key, inner_key, default_value=None):
        return self.container.get(key, {}).get(inner_key, default_value)

    def set(self, key, inner_key, value):
        self.container[key][inner_key] = value

    def get_inner(self, key):
        return self.container[key]

    def get_outer(self, inner_key, default_value=None):
        outer_dict = {key: default_value for key in self.container}
        for key, inner_dict in self.container.items():
            for _inner_key, value in inner_dict.items():
                if _inner_key == inner_key:
                    outer_dict[key] = value
        return outer_dict

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            return self.set(key[0], key[1], value)
        elif isinstance(value, dict):
            self.container[key] = value
        else:
            raise ValueError("Key in __setitem__ must be a tuple")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.get(item[0], item[1])
        else:
            return self.get_inner(item)

    def items(self, default_value=None):
        for key in self.keys():
            yield key, self.get(*key, default_value=default_value)

    def keys(self):
        for key, inner_dict in self.container.items():
            for inner_key in inner_dict:
                yield key, inner_key

    def to_dataframe(
            self,
            *,
            outer_column: str = "outer",
            inner_column: str = "inner",
            value_column: str = "value",
            default_value=None,
    ) -> pd.DataFrame:
        all_keys = list(self.keys())
        all_outer_keys = list(set(k[0] for k in all_keys))
        all_inner_keys = list(set(k[1] for k in all_keys))
        df_dict = {
            outer_column: [],
            inner_column: [],
            value_column: [],
        }
        for ok in all_outer_keys:
            for ik in all_inner_keys:
                df_dict[outer_column].append(ok)
                df_dict[inner_column].append(ik)
                df_dict[value_column].append(self.get(ok, ik, default_value))
        return pd.DataFrame(df_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},outer_keys={list(self.container.keys())})"

    def save_item(self, filepath: str, *key):
        import pickle

        item = self.__getitem__(key)
        if item is None:
            return
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(item, f)
        return self

    def load_item(self, filepath, *key):
        import pickle

        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        if not os.path.isfile(filepath):
            return self
        with open(filepath, "rb") as f:
            item = pickle.load(f)
        self.__setitem__(key, item)
        return self

    def __contains__(self, item):
        if isinstance(item, tuple):
            return self.get(item[0], item[1]) is not None
        else:
            return item in self.container

    def save_item_to_txt(self, filepath: str, *key):
        if not filepath.endswith(".txt"):
            filepath += ".txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        item = self.__getitem__(key)
        with open(filepath, "w") as f:
            f.write(str(item))


class MetricsContainer:
    ACCURACY_KEY = "Accuracy"
    F1_KEY = "F1-Score"
    PRECISION_KEY = "Precision"
    RECALL_KEY = "Recall"
    available_metrics = {
        ACCURACY_KEY: accuracy_score,
        F1_KEY: partial(f1_score, average="weighted"),
        PRECISION_KEY: partial(precision_score, average="weighted"),
        RECALL_KEY: partial(recall_score, average="weighted"),
    }

    def __init__(
            self,
            metrics: Optional[List[str]] = None,
            *,
            pre_name: str = "",
            post_name: str = "",
            name_separator: str = "_",
    ):
        self.metrics = metrics or list(self.available_metrics.keys())
        self.pre_name = pre_name
        self.post_name = post_name
        self.name_separator = name_separator
        if pre_name:
            pre_name += name_separator
        if post_name:
            post_name = name_separator + post_name
        self.metrics_names = [
            f"{pre_name}{name}{post_name}" for name in self.metrics
        ]
        self.containers = {
            metric: KPredictorContainer(metric_name)
            for metric, metric_name in zip(self.metrics, self.metrics_names)
        }

    @property
    def containers_list(self):
        return list(self.containers.values())

    def get_is_metrics_all_computed(self, key, inner_key):
        return all(self.get(metric, key, inner_key, None) is not None for metric in self.metrics)

    def get(self, metric: str, key, inner_key, default_value=None):
        return self.containers[metric].get(key, inner_key, default_value=default_value)

    def set(self, metric: str, key, inner_key, value):
        self.containers[metric].set(key, inner_key, value)

    def get_metric(self, metric: str):
        return self.containers[metric]

    def compute_metrics(self, y_true, y_pred, key, inner_key, **kwargs):
        for metric in self.metrics:
            self.set(metric, key, inner_key, self.available_metrics[metric](y_true, y_pred))
        return self.containers

    def get_item_metrics(self, *key, default_value=None):
        return {
            metric: self.containers[metric].__getitem__(key)
            if self.containers[metric].__contains__(key) else default_value
            for metric in self.metrics
        }

    def save_item_metrics(self, filepath: str, *key):
        import pickle

        to_save = self.get_item_metrics(*key)
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(to_save, f)
        return self

    def load_item_metrics(self, filepath: str, *key):
        import pickle

        if not filepath.endswith(".pkl"):
            filepath += ".pkl"
        if not os.path.isfile(filepath):
            return self
        with open(filepath, "rb") as f:
            item = pickle.load(f)
        for metric, value in item.items():
            self.set(metric, *key, value)
        return self

    def save_item_metrics_to_txt(self, filepath: str, *key):
        if not filepath.endswith(".txt"):
            filepath += ".txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        item = self.get_item_metrics(*key)
        with open(filepath, "w") as f:
            f.write(str(item))

