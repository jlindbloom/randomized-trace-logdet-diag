import numpy as np

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator
    
    
    
def explicit_trace_probe(A):
    """Computes the trace of A using an explicit probe. Requires exactly n matvecs with A.
    """

    # Setup
    n = A.shape[0]
    diagonal = np.zeros(n)
    
    # Handle CuPy
    if CUPY_INSTALLED:
        if isinstance(A, CuPyLinearOperator):
            xp = cp
        else:
            xp = np
    else:
        xp = np
    
    for j in range(n):

        # jth column of the identity
        w = xp.zeros(n)
        w[j] = 1.0

        # Compute w^T A w
        diagonal[j] = w.T @ (A @ w)

    trace = xp.sum(diagonal)

    return trace





