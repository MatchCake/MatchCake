from typing import Optional, Dict, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .visualizer import Visualizer
from ..cross_validation import CrossValidationOutput
from .mpl_rcparams import MPL_RC_DEFAULT_PARAMS


class CrossValidationVisualizer(Visualizer):
    def __init__(
            self,
            cross_validation_output: CrossValidationOutput
    ):
        mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)
        self.cvo = cross_validation_output

    def plot(
            self,
            *,
            ax: Optional[plt.Axes] = None,
            score_name: str = "Score",
            score_split_name: str = "Splits",
            score_name_map: Optional[Dict[str, str]] = None,
            palette: Optional[Union[str, list, Dict]] = "colorblind",
    ):
        melted_df = self.cvo.results_df.melt(
            id_vars=[self.cvo.estimator_name_key],
            value_vars=self.cvo.score_columns,
            var_name=score_split_name,
            value_name=score_name,
        )
        if score_name_map is not None:
            melted_df[score_split_name] = melted_df[score_split_name].map(score_name_map)
        ax = sns.violinplot(
            data=melted_df,
            x=self.cvo.estimator_name_key,
            y=score_name,
            hue=score_split_name,
            split=True,
            inner="quart",
            cut=0,
            palette=palette,
            ax=ax,
        )
        return ax


