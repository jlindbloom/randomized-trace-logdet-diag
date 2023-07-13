import numpy as np
from scipy.linalg import cho_factor as scipy_chol_fac
from scipy.linalg import cho_solve as scipy_chol_solve



def naive_diaginv(A, sample_size=1000, method="cholesky"):
    """Naive unbiased estimator for the diagonal of an inverse matrix, see [5]. A must be SPD.
    """

    valid_methods = ["cholesky"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    # Get shape
    n = A.shape[0]

    # Setup
    diaginv_estimate = np.zeros(n)
    tk = np.zeros(n)
    qk = np.zeros(n)

    if method == "cholesky":
        chol = scipy_chol_fac(A)
    else:
        raise NotImplementedError

    for j in range(sample_size):
        
        # Draw random vector
        vk = np.random.choice([-1, 1], size=n)

        # Update tk
        if method == "cholesky":
            Ainv_vk = scipy_chol_solve(chol, vk)
        else:
            raise NotImplementedError
    
        tk = tk + (( Ainv_vk ) * vk)

        # Update qk
        qk = qk + (vk*vk)

        # Update diag_estimate
        diaginv_estimate = tk / qk

    return diaginv_estimate




