import numpy as np



def explicit_trace_probe(A):
    """Computes the trace of A using an explicit probe. Requires exactly n matvecs with A.
    """

    # Setup
    n = A.shape[0]
    diagonal = np.zeros(n)

    for j in range(n):

        # jth column of the identity
        w = np.zeros(n)
        w[j] = 1.0

        # Compute w^T A w
        diagonal[j] = w.T @ (A @ w)

    trace = np.sum(diagonal)

    return trace









