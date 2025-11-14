import numpy as np

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_I = np.eye(2)
CLIFFORD_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
CLIFFORD_S = np.array([[1, 0], [0, 1j]])
