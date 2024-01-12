import numpy as np
from typing import Any
from numpy.typing import NDArray


def positive_definite_matrix(A: NDArray[Any]):
    """
    Checks if a matrix is positive definite
    :param A: Matrix to check
    """
    if np.allclose(A, A.T):  # Check if the matrix is symmetric
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
