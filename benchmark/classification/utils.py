import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

MPL_RC_DEFAULT_PARAMS = {
    "font.size": 18,
    "legend.fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "xtick.direction": "in",
    "xtick.major.size": 6,
    "xtick.major.width": 3,
    'xtick.minor.visible': True,
    'xtick.minor.size': 3.0,
    'xtick.minor.width': 2.0,
    "ytick.direction": "in",
    "ytick.major.width": 3,
    "ytick.major.size": 6,
    'ytick.minor.visible': True,
    'ytick.minor.size': 3.0,
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
}

MPL_RC_BIG_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 22,
    "legend.fontsize": 20,
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
