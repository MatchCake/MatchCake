from typing import Any, Callable, Optional

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
        *,
        x: Optional[np.ndarray] = None,
        x_reduced: Optional[np.ndarray] = None,
        x_mesh: Optional[np.ndarray] = None,
        reducer: Optional[Any] = None,
        transform: Optional[Callable] = None,
        inverse_transform: Optional[Callable] = None,
        n_pts: int = 1_000,
        seed: Optional[int] = 0,
        **kwargs,
    ):
        self.x = x
        self.reducer = reducer
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.n_pts = n_pts
        self.seed = seed
        self.kwargs = kwargs

        self.x_reduced = x_reduced
        self.x_mesh = x_mesh

    def gather_transforms(self, **kwargs):
        """
        If a transform and an inverse_transform functions are given, they will be returned. Otherwise, the transform and
        inverse_transform will be inferred from the given reducer. If the reducer is None, then it will be
        initialized as a PCA with 2 components.
        """
        need_reducer = (self.transform is None) or (self.inverse_transform is None)
        if not need_reducer:
            return self.transform, self.inverse_transform
        kwargs = {**self.kwargs, **kwargs}
        if self.transform is not None:
            assert self.inverse_transform is not None, "inverse_transform must be given if transform is given."

        if need_reducer:
            if self.reducer is None:
                self.reducer = "pca"
            if isinstance(self.reducer, str):
                n_jobs = kwargs.get("n_jobs", max(0, psutil.cpu_count() - 2))
                if self.reducer.lower() == "pca":
                    self.reducer = decomposition.PCA(n_components=2, random_state=self.seed)
                elif self.reducer.lower() == "umap":
                    import umap

                    self.reducer = umap.UMAP(n_components=2, transform_seed=self.seed, n_jobs=n_jobs)
                else:
                    raise ValueError(f"Unknown reducer: {self.reducer}")
            if kwargs.get("check_estimators", True):
                check_estimator(self.reducer)
            self.reducer.fit(self.x)
            self.transform = self.reducer.transform
            self.inverse_transform = self.reducer.inverse_transform
        return self.transform, self.inverse_transform

    def compute_x_reduced(self, **kwargs):
        if self.x_reduced is not None:
            return self.x_reduced

        kwargs = {**self.kwargs, **kwargs}
        if self.x.shape[-1] == 2:
            self.x_reduced = self.x
        else:
            self.gather_transforms(**kwargs)
            self.x_reduced = self.transform(self.x)

        if self.x_reduced.ndim != 2:
            raise ValueError(f"x_reduced.ndim = {self.x_reduced.ndim} != 2. The given reducer does not reduce to 2D.")
        return self.x_reduced

    def compute_x_mesh(self, **kwargs):
        if self.x_mesh is not None:
            self.n_pts = self.x_mesh.shape[0]
            return self.x_mesh
        kwargs = {**self.kwargs, **kwargs}
        x_min, x_max = self.x_reduced[:, 0].min() - 1, self.x_reduced[:, 0].max() + 1
        y_min, y_max = self.x_reduced[:, 1].min() - 1, self.x_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, num=int(np.sqrt(self.n_pts))),
            np.linspace(y_min, y_max, num=int(np.sqrt(self.n_pts))),
        )

        x_mesh_reduced = np.c_[xx.ravel(), yy.ravel()]
        if self.inverse_transform is None:
            self.x_mesh = x_mesh_reduced
        else:
            self.gather_transforms(**kwargs)
            self.x_mesh = self.inverse_transform(x_mesh_reduced)
        self.n_pts = self.x_mesh.shape[0]
        return self.x_mesh

    def plot_2d_decision_boundaries(
        self,
        *,
        y: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        y_pred: Optional[np.ndarray] = None,
        **kwargs,
    ):
        kwargs = {**self.kwargs, **kwargs}
        axis_label_cue = None
        if self.reducer is None:
            self.reducer = "pca"
        if isinstance(self.reducer, str):
            axis_label_cue = self.reducer.upper()

        if self.x_reduced is None:
            self.x_reduced = self.compute_x_reduced(**kwargs)
        predict_func = kwargs.get("predict_func", getattr(model, "predict", None))
        if y_pred is None and self.x_mesh is None:
            self.x_mesh = self.compute_x_mesh(**kwargs)
        if y_pred is None:
            if predict_func is None:
                raise ValueError("Either y_pred or predict_func must be given.")
            y_pred = predict_func(self.x_mesh)
        x_min, x_max = self.x_reduced[:, 0].min() - 1, self.x_reduced[:, 0].max() + 1
        y_min, y_max = self.x_reduced[:, 1].min() - 1, self.x_reduced[:, 1].max() + 1
        side_length = int(np.sqrt(y_pred.shape[0]))
        y_mesh = y_pred.reshape((side_length, side_length))

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
                self.x_reduced[:, 0],
                self.x_reduced[:, 1],
                c=[cmap(i) for i in y],
                edgecolor="k",
                linewidths=kwargs.get("linewidths", 1.5),
                s=kwargs.get("s", 100),
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # legend_labels = getattr(dataset, "target_names", list(range(N_labels)))
        legend_labels = kwargs.get("legend_labels", None)
        if legend_labels is None:
            legend_labels = list(range(n_labels))
        patches = []
        for i, legend in enumerate(legend_labels):
            patch = matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=cmap(i),
                markersize=kwargs.get("markersize", 10),
                markeredgecolor="k",
                label=legend,
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

        if kwargs.get("show", False):
            plt.show()
        return fig, ax, y_pred
