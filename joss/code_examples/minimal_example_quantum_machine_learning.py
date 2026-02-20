import matplotlib as mpl
from matchcake.ml import CrossValidation, CrossValidationVisualizer
from matchcake.ml.kernels import FermionicPQCKernel
from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel
from matchcake.ml.visualisation import ClassificationVisualizer
from matchcake.ml.visualisation.mpl_rcparams import MPL_RC_DEFAULT_PARAMS
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)

# Load the iris dataset
dataset = datasets.load_iris(as_frame=True)
X, y = dataset.data, dataset.target

# Define estimators for comparison, using different kernels
n_qubits = 8
estimators = {
    "Fermionic PQC": Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            # Example of kernel without alignment, which is the default behaviour
            ("kernel", FermionicPQCKernel(n_qubits=n_qubits, rotations="X,Z")),
            ("classifier", SVC(kernel="precomputed")),
        ]
    ),
    "Fermionic Linear": Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            # Example of kernel with alignment, which improves the performance
            ("kernel", LinearNIFKernel(n_qubits=n_qubits, alignment=True)),
            ("classifier", SVC(kernel="precomputed")),
        ]
    ),
}

# Run cross-validation
cvo = CrossValidation(estimators, X, y).run()

# Create subplots for visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# Visualize the classification boundaries for a selected model and fold
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
viz = ClassificationVisualizer(x=X, n_pts=25_000)
viz.plot_2d_decision_boundaries(
    model=estimators["Fermionic Linear"].fit(x_train, y_train),
    title="",
    legend_labels=dataset.target_names,
    y=y,
    fig=fig,
    ax=axes[0],
    legend_loc="lower right",
)
# Visualize cross-validation results with violin plots
cv_viz = CrossValidationVisualizer(cvo)
cv_viz.plot(
    ax=axes[1],
    score_name="Accuracy",
    estimator_name_key="Kernel",
    score_name_map={"train_score": "Train", "test_score": "Test"},
    palette="pastel",
)

if __name__ == "__main__":
    axes[1].set_ylim(axes[1].get_ylim()[0], 1.0)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("../images/minimal_example_quantum_machine_learning.svg", dpi=900)
    plt.show()
