import numpy as np
from scipy.linalg import cho_factor as scipy_chol_fac
from scipy.linalg import cho_solve as scipy_chol_solve



def explicit_diaginv_probe(A, method="cholesky"):
    """Computes the diagonal of inv(A) using an explicit probe. A must be SPD.
    """

    valid_methods = ["cholesky"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    # Setup
    n = A.shape[0]
    diagonal_inv = np.zeros(n)

    if method == "cholesky":
        chol = scipy_chol_fac(A)
    else:
        raise NotImplementedError

    for j in range(n):

        # jth column of the identity
        w = np.zeros(n)
        w[j] = 1.0

        # Compute w^T inv(A) w
        if method == "cholesky":
            Ainv_w = scipy_chol_solve(chol, w)
        else:
            raise NotImplementedError

        diagonal_inv[j] = w.T @ Ainv_w

    return diagonal_inv
