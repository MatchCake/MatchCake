from .k_qubits_by_k_qubits_sampling import KQubitsByKQubitsSampling


class QubitByQubitSampling(KQubitsByKQubitsSampling):
    """Autoregressive sampler drawing a single qubit per step (``K = 1``)."""

    NAME = "QubitByQubitSampling"
    K: int = 1
