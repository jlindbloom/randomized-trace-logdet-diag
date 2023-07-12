import numpy as np



def logdet_via_cholesky(A, banded_cholesky=False):
    """Computes logdet(A) using the Cholesky method. A must be SPD."""

    if not banded_cholesky:
        chol = np.linalg.cholesky(A)
    else:
        raise NotImplementedError

    return 2*np.sum(np.log(np.diag(chol)))
