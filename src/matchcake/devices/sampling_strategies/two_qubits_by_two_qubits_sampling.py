from .k_qubits_by_k_qubits_sampling import KQubitsByKQubitsSampling


class TwoQubitsByTwoQubitsSampling(KQubitsByKQubitsSampling):
    NAME = "2QubitBy2QubitSampling"
    K: int = 2
