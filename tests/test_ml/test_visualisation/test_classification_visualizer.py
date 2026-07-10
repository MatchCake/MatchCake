from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
from sklearn.decomposition import PCA

from matchcake.ml.visualisation.classification_visualizer import ClassificationVisualizer

from ...configs import TEST_SEED, set_seed

matplotlib.use("Agg")


class TestClassificationVisualizer:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)
        rng = np.random.default_rng(TEST_SEED)
        cls.x_2d = rng.standard_normal((20, 2))
        cls.x_4d = rng.standard_normal((20, 4))
        cls.y = (rng.random(20) > 0.5).astype(int)

    def test_init(self):
        vis = ClassificationVisualizer(x=self.x_2d)
        assert vis.x is self.x_2d
        assert vis.reducer is None
        assert vis.transform is None
        assert vis.inverse_transform is None
        assert vis.n_pts == 1_000
        assert vis.seed == 0
        assert vis.x_reduced is None
        assert vis.x_mesh is None

    def test_gather_transforms_with_callables(self):
        def transform(x):
            return x

        def inverse_transform(x):
            return x

        vis = ClassificationVisualizer(x=self.x_2d, transform=transform, inverse_transform=inverse_transform)
        t, inv = vis.gather_transforms()
        assert t is transform
        assert inv is inverse_transform

    def test_gather_transforms_uses_pca_by_default(self):
        vis = ClassificationVisualizer(x=self.x_4d)
        vis.gather_transforms(check_estimators=False)
        assert vis.transform is not None
        assert vis.inverse_transform is not None

    def test_gather_transforms_pca_string(self):
        vis = ClassificationVisualizer(x=self.x_4d, reducer="pca")
        vis.gather_transforms(check_estimators=False)
        assert isinstance(vis.reducer, PCA)

    def test_gather_transforms_unknown_reducer_raises(self):
        vis = ClassificationVisualizer(x=self.x_4d, reducer="unknown_reducer")
        with pytest.raises(ValueError, match="Unknown reducer"):
            vis.gather_transforms(check_estimators=False)

    def test_compute_x_reduced_already_set(self):
        x_reduced = self.x_2d.copy()
        vis = ClassificationVisualizer(x=self.x_4d, x_reduced=x_reduced)
        result = vis.compute_x_reduced()
        assert result is x_reduced

    def test_compute_x_reduced_2d_input_no_reduction(self):
        vis = ClassificationVisualizer(x=self.x_2d)
        result = vis.compute_x_reduced()
        np.testing.assert_array_equal(result, self.x_2d)

    def test_compute_x_reduced_4d_input_reduces_to_2d(self):
        vis = ClassificationVisualizer(x=self.x_4d)
        result = vis.compute_x_reduced(check_estimators=False)
        assert result.shape == (len(self.x_4d), 2)

    def test_compute_x_mesh_already_set(self):
        x_mesh = self.x_2d.copy()
        vis = ClassificationVisualizer(x=self.x_2d, x_reduced=self.x_2d, x_mesh=x_mesh)
        result = vis.compute_x_mesh()
        assert result is x_mesh

    def test_compute_x_mesh_no_inverse_transform(self):
        vis = ClassificationVisualizer(x=self.x_2d, x_reduced=self.x_2d)
        result = vis.compute_x_mesh()
        assert result.shape[-1] == 2

    def test_compute_x_mesh_with_inverse_transform(self):
        vis = ClassificationVisualizer(
            x=self.x_4d,
            transform=lambda x: x[:, :2],
            inverse_transform=lambda x: np.hstack([x, np.zeros((len(x), 2))]),
        )
        vis.compute_x_reduced()
        result = vis.compute_x_mesh()
        assert result.shape[-1] == 4

    def test_gather_transforms_asserts_inverse_when_transform_given(self):
        vis = ClassificationVisualizer(x=self.x_4d, transform=lambda x: x[:, :2])
        with pytest.raises(AssertionError):
            vis.gather_transforms(check_estimators=False)

    def test_compute_x_reduced_raises_for_non_2d_reducer(self):
        def bad_transform(x):
            return x[:, :, np.newaxis]

        vis = ClassificationVisualizer(
            x=self.x_4d,
            transform=bad_transform,
            inverse_transform=lambda x: x,
        )
        with pytest.raises(ValueError, match="x_reduced.ndim"):
            vis.compute_x_reduced()

    @pytest.mark.parametrize("with_y", [True, False])
    def test_plot_2d_decision_boundaries(self, with_y):
        vis = ClassificationVisualizer(x=self.x_2d)
        side = 10
        y_pred = np.tile(np.array([0, 1]), side * side // 2)
        fig, ax, returned_pred = vis.plot_2d_decision_boundaries(
            y=self.y if with_y else None,
            y_pred=y_pred,
        )
        assert fig is not None
        assert ax is not None
        np.testing.assert_array_equal(returned_pred, y_pred)

    def test_plot_2d_decision_boundaries_with_predict_func(self):
        side = 10
        y_pred_full = np.tile(np.array([0, 1]), side * side // 2)
        vis = ClassificationVisualizer(x=self.x_2d, x_reduced=self.x_2d)
        vis.compute_x_mesh()
        fig, ax, returned_pred = vis.plot_2d_decision_boundaries(
            predict_func=lambda x: y_pred_full[: len(x)],
        )
        assert fig is not None

    def test_plot_2d_decision_boundaries_no_predict_raises(self):
        vis = ClassificationVisualizer(x=self.x_2d, x_reduced=self.x_2d)
        with pytest.raises(ValueError, match="Either y_pred or predict_func"):
            vis.plot_2d_decision_boundaries()

    def test_plot_2d_decision_boundaries_hide_ticks_false(self):
        vis = ClassificationVisualizer(x=self.x_2d)
        side = 10
        y_pred = np.tile(np.array([0, 1]), side * side // 2)
        fig, ax, _ = vis.plot_2d_decision_boundaries(y_pred=y_pred, hide_ticks=False)
        assert fig is not None

    def test_plot_2d_decision_boundaries_axis_name_none(self):
        pca = PCA(n_components=2)
        pca.fit(self.x_4d)
        vis = ClassificationVisualizer(
            x=self.x_4d,
            reducer=pca,
            transform=pca.transform,
            inverse_transform=pca.inverse_transform,
        )
        vis.compute_x_reduced()
        side = 10
        y_pred = np.tile(np.array([0, 1]), side * side // 2)
        fig, ax, _ = vis.plot_2d_decision_boundaries(y_pred=y_pred)
        assert ax.get_xlabel() != ""

    def test_plot_2d_decision_boundaries_show(self):
        vis = ClassificationVisualizer(x=self.x_2d)
        side = 10
        y_pred = np.tile(np.array([0, 1]), side * side // 2)
        with patch("matplotlib.pyplot.show"):
            vis.plot_2d_decision_boundaries(y_pred=y_pred, show=True)
