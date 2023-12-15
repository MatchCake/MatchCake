from typing import Optional, Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import decomposition
from sklearn.utils.estimator_checks import check_estimator


class Visualizer:
    pass


class ClassificationVisualizer(Visualizer):
    def __init__(
            self,
    ):
        pass
    
    @classmethod
    def gather_transforms(
            cls,
            *,
            x: np.ndarray,
            reducer: Optional[Any] = None,
            transform: Optional[Callable] = None,
            inverse_transform: Optional[Callable] = None,
            seed: Optional[int] = 0,
            **kwargs
    ):
        """
        If a transform and an inverse_transform functions are given, they will be returned. Otherwise, the transform and
        inverse_transform will be inferred from the given reducer. If the reducer is None, then it will be
        initialized as a PCA with 2 components.
        """
        need_reducer = (transform is None) or (inverse_transform is None)
        
        if transform is not None:
            assert inverse_transform is not None, "inverse_transform must be given if transform is given."
        
        if need_reducer:
            if reducer is None:
                reducer = "pca"
            if isinstance(reducer, str):
                n_jobs = kwargs.get("n_jobs", max(0, psutil.cpu_count() - 2))
                if reducer.lower() == "pca":
                    reducer = decomposition.PCA(n_components=2, random_state=seed)
                elif reducer.lower() == "umap":
                    import umap
                    reducer = umap.UMAP(n_components=2, transform_seed=seed, n_jobs=n_jobs)
                else:
                    raise ValueError(f"Unknown reducer: {reducer}")
            if kwargs.get("check_estimators", True):
                check_estimator(reducer)
            reducer.fit(x)
            transform = reducer.transform
            inverse_transform = reducer.inverse_transform
        return transform, inverse_transform, need_reducer
    
    @classmethod
    def plot_2d_decision_boundaries(
            cls,
            model: Any,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            *,
            reducer: Optional[Any] = None,
            transform: Optional[Callable] = None,
            inverse_transform: Optional[Callable] = None,
            n_pts: int = 1_000,
            seed: Optional[int] = 0,
            **kwargs
    ):
        axis_label_cue = None
        if reducer is None:
            reducer = "pca"
        if isinstance(reducer, str):
            axis_label_cue = reducer.upper()
        
        transform, inverse_transform, is_default = cls.gather_transforms(
            x=X,
            reducer=reducer,
            transform=transform,
            inverse_transform=inverse_transform,
            seed=seed,
            **kwargs
        )
        need_transform = (not is_default) or (X.shape[-1] != 2)
        
        if kwargs.get("check_estimators", True):
            check_estimator(model)
        
        if need_transform:
            x_reduced = transform(X)
        else:
            x_reduced = X
        
        if x_reduced.ndim != 2:
            raise ValueError(f"x_reduced.ndim = {x_reduced.ndim} != 2. The given reducer does not reduce to 2D.")
        
        x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
        y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, num=int(np.sqrt(n_pts))),
            np.linspace(y_min, y_max, num=int(np.sqrt(n_pts)))
        )
        
        x_reduced_mesh = np.c_[xx.ravel(), yy.ravel()]
        if need_transform:
            x_mesh = inverse_transform(x_reduced_mesh)
        else:
            x_mesh = x_reduced_mesh
        y_pred = model.predict(x_mesh)
        y_mesh = y_pred.reshape(xx.shape)
        
        fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(14, 10))
        
        if y is None:
            n_labels = len(np.unique(y_pred))
        else:
            n_labels = len(np.unique(y))
        base_cmap = sns.color_palette(palette=None, n_colors=n_labels, as_cmap=False)
        cmap = ListedColormap([base_cmap[i] for i in range(n_labels)])
        ax.imshow(
            y_mesh,
            cmap=cmap,
            vmin=0,
            vmax=n_labels - 1,
            alpha=kwargs.get("alpha", 0.8),
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect=kwargs.get("aspect", "auto"),
            interpolation=kwargs.get("interpolation", "antialiased"),
        )
        if y is not None:
            scatter = ax.scatter(
                x_reduced[:, 0],
                x_reduced[:, 1],
                c=[cmap(i) for i in y],
                edgecolor='k',
                linewidths=kwargs.get("linewidths", 1.5),
                s=kwargs.get("s", 100),
            )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        # legend_labels = getattr(dataset, "target_names", list(range(N_labels)))
        legend_labels = kwargs.get("legend_labels", None)
        if legend_labels is None:
            legend_labels = list(range(n_labels))
        patches = []
        for i, legend in enumerate(legend_labels):
            patch = matplotlib.lines.Line2D(
                [0], [0], marker='o', linestyle='None', markerfacecolor=cmap(i),
                markersize=kwargs.get("markersize", 10), markeredgecolor='k', label=legend
            )
            patches.append(patch)
        
        ax.set_title(
            kwargs.get("title", "Decision boundaries in the reduced space."),
            fontsize=kwargs.get("fontsize", 18),
        )
        
        ax.legend(handles=patches, fontsize=kwargs.get("fontsize", 18))
        if kwargs.get("hide_ticks", True):
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.minorticks_on()
        axis_name = kwargs.get("axis_name", axis_label_cue)
        if axis_name is None:
            axis_name = "Reduced"
        x_label = kwargs.get("x_label", f"{axis_name} 1")
        y_label = kwargs.get("y_label", f"{axis_name} 2")
        ax.set_xlabel(x_label, fontsize=kwargs.get("fontsize", 18))
        ax.set_ylabel(y_label, fontsize=kwargs.get("fontsize", 18))
        
        kwargs["output"] = dict(
            x_reduced=x_reduced,
            x_mesh=x_mesh,
            y_mesh=y_mesh,
            y_pred=y_pred,
        )
        
        if kwargs.get("show", False):
            plt.show()
        return fig, ax
