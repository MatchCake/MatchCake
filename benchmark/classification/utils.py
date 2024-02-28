import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

MPL_RC_DEFAULT_PARAMS = {
    "font.size": 18,
    "legend.fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
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
}}

MPL_RC_SMALL_FONT_PARAMS = {**MPL_RC_DEFAULT_PARAMS, **{
    "font.size": 12,
    "legend.fontsize": 10,
}}
mStyles = [
    ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d",
    "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
]


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


def exponential2_fit(x, y, **kwargs):
    """
    Fit data to an exponential function of the form y = c + a * 2^(b * x)

    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    x_lbl = kwargs.pop('x_lbl', 'x')
    popt, pcov = curve_fit(exponential2, x, y, **kwargs)
    label = f"~$O({popt[0]*2:.4f}^{{ {popt[1]:.4f}{x_lbl} }})$"
    r2 = r2_score(y, exponential2(x, *popt))
    return {
        'popt': popt,
        'pcov': pcov,
        'r_squared': r2,
        'label': label,
    }

