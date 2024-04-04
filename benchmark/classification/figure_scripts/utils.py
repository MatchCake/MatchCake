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
