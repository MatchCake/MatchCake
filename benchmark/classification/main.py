import time
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import umap

from kernels import ClassicalKernel, PennylaneQuantumKernel, NIFKernel
try:
    import msim
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import msim
from msim.ml import ClassificationVisualizer

if __name__ == '__main__':
    # dataset = datasets.load_breast_cancer(as_frame=True)
    # dataset = datasets.load_iris(as_frame=True)
    dataset = datasets.make_classification(
        n_samples=100,
        n_features=4,
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    if isinstance(dataset, tuple):
        X, y = dataset
    elif isinstance(dataset, dict):
        X = dataset["data"]
        y = dataset["target"]
    elif isinstance(dataset, pd.DataFrame):
        X = dataset.data
        y = dataset.target
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")
    
    # X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    # y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1).astype(int)
    print(f"(N Samples, N features): {X.shape}")
    print(f"Classes: {set(np.unique(y))}, labels: {getattr(dataset, 'target_names', set(np.unique(y)))}")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print(f"N train samples: {x_train.shape[0]}, N test samples: {x_test.shape[0]}")
    
    embedding_size = X.shape[-1]
    print(f"Embedding size: {embedding_size}")
    
    rn_state = np.random.RandomState(seed=0)
    rn_embed_matrix = rn_state.randn(X.shape[-1], embedding_size)
    
    clas_kernel = ClassicalKernel(
        embedding_dim=embedding_size,
        metric="rbf",
        # encoder_matrix=rn_embed_matrix,
        seed=0
    ).fit(X, y)
    pennylane_kernel = PennylaneQuantumKernel(
        embedding_dim=embedding_size,
        seed=0,
        # encoder_matrix=rn_embed_matrix,
        shots=1,
        # nb_workers=max(0, psutil.cpu_count(logical=False) - 2),
        interface="auto",
    ).fit(X, y)
    nif_kernel = NIFKernel(
        embedding_dim=embedding_size,
        seed=0,
        # encoder_matrix=rn_embed_matrix,
        # nb_workers=max(0, psutil.cpu_count(logical=False) - 2),
        interface="auto",
    ).fit(X, y)
    
    clas_model = svm.SVC(kernel=clas_kernel.kernel, random_state=0)
    pennylane_model = svm.SVC(kernel=pennylane_kernel.kernel, random_state=0)
    nif_model = svm.SVC(kernel=nif_kernel.kernel, random_state=0)
    
    models = {
        # "classical": clas_model,
        # "pennylane": pennylane_model,
        "nif": nif_model,
    }
    n_plots = len(models)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
    axes = np.ravel(np.asarray([axes]))
    for i, (m_name, model) in enumerate(models.items()):
        fit_start_time = time.time()
        model.fit(x_train, y_train)
        fit_end_time = time.time()
        fit_time = fit_end_time - fit_start_time
        accuracy = model.score(x_test, y_test)
        plot_start_time = time.time()
        fig, ax = ClassificationVisualizer.plot_2d_decision_boundaries(
            model=model,
            X=X, y=y,
            # reducer=decomposition.PCA(n_components=2, random_state=0),
            # reducer=umap.UMAP(n_components=2, transform_seed=0, n_jobs=max(0, psutil.cpu_count() - 2)),
            check_estimators=False,
            n_pts=1_000,
            title=f"Decision boundaries in the reduced space.",
            legend_labels=getattr(dataset, "target_names", None),
            # axis_name="RN",
            fig=fig, ax=axes[i],
            interpolation="nearest",
        )
        ax.set_title(f"{m_name} accuracy: {accuracy * 100:.2f}%")
        plot_end_time = time.time()
        plot_time = plot_end_time - plot_start_time
        print(f"{m_name} test accuracy: {accuracy * 100 :.4f}%, {fit_time = :.5f} [s], {plot_time = :.5f} [s]")
    
    plt.show()

