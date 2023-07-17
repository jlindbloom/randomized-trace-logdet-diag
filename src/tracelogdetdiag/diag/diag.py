import numpy as np

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator
    
    

def naive_diag(A, sample_size=1000):
    """Naive unbiased estimator for the diagonal of a matrix, see [5]. A must be SPSD.
    """

    # Get shape
    n = A.shape[0]
    
    # Handle CuPy
    if CUPY_INSTALLED:
        if isinstance(A, CuPyLinearOperator):
            xp = cp
        else:
            xp = np
    else:
        xp = np

    diag_estimate = xp.zeros(n)
    tk = xp.zeros(n)
    qk = xp.zeros(n)

    for j in range(sample_size):
        
        # Draw random vector
        vk = xp.random.choice([-1, 1], size=n)

        # Update tk
        tk = tk + ((A @ vk) * vk)

        # Update qk
        qk = qk + (vk*vk)

        # Update diag_estimate
        diag_estimate = tk / qk

    return diag_estimate











