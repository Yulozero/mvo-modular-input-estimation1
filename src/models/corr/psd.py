import numpy as np


class PSDProjector:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def project(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        matrix = (matrix + matrix.T) / 2.0

        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.asarray(eigvals, dtype=float)
        eigvals = np.clip(eigvals, self.eps, None)

        projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
        projected = np.asarray(projected, dtype=float)
        projected = (projected + projected.T) / 2.0
        return projected
