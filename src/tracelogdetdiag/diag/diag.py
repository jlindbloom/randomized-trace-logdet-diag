import numpy as np



def naive_diag(A, sample_size=1000):
    """Naive unbiased estimator for the diagonal of a matrix, see [5]. A must be SPSD.
    """

    # Get shape
    n = A.shape[0]

    diag_estimate = np.zeros(n)
    tk = np.zeros(n)
    qk = np.zeros(n)

    for j in range(sample_size):
        
        # Draw random vector
        vk = np.random.choice([-1, 1], size=n)

        # Update tk
        tk = tk + ((A @ vk) * vk)

        # Update qk
        qk = qk + (vk*vk)

        # Update diag_estimate
        diag_estimate = tk / qk

    return diag_estimate











