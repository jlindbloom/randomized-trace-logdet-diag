import numpy as np
from scipy.linalg import qr as scipy_qr



def hutchinson_trace(A, sample_size=100, block_size=20, method="rademacher", exact_sample_size=False):
    """Computes the Hutchinson randomized estimator of tr(A). A must be SPSD.
    
    Here we compute the estimator with sample_size using blocks of samples of size ceil(sample_size/block_size).
    This helps control memory usage vs. vectorization. We don't throw away any samples, so the estimator may be
    computed with a slightly larger sample size than specified, unless exact_sample_size=True.
    """

    # Get shape
    n = A.shape[0]

    valid_methods = ["standard_gaussian", "rademacher"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    # Handle blocks
    n_blocks = int(np.ceil(sample_size/block_size))
    extra_samples = (block_size*n_blocks) - sample_size

    block_sums = []
    for j in range(n_blocks):

        # Draw random block of vectors
        if method == "standard_gaussian":
            w = np.random.normal(size=(n, block_size))
        elif method == "rademacher":
            w = np.random.choice([-1, 1], size=(n, block_size))
        else:
            raise NotImplementedError
        
        if (j == n_blocks - 1) and (exact_sample_size == True):
            w = w[:,:-extra_samples]
        
        # Append block sum
        block_sum = np.sum( ( (A.T @ w).T * w.T ).sum(axis=1)  )
        block_sums.append(block_sum)

    tot_sum = np.sum(block_sums)
    if exact_sample_size:
        estimate = tot_sum/sample_size
    else:
        estimate = tot_sum/(block_size*n_blocks)

    return estimate



def hutchinson_epsilon_delta_trace(A, epsilon=0.05, delta=0.05, method="rademacher", block_size=20):
    """Computes an (epsilon, delta)-estimator of trace(A). A must be SPSD. This uses lower-bounds from the literature to pick a sample size 
    for the Hutchinson estimator \hat{tr}(A) such that | \hat{tr}(A) - tr(A) | < epsilon*tr(A) with probability greater than 1 - delta."""
    
    valid_methods = ["standard_gaussian", "rademacher"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    c = (1.0/(epsilon**2))*np.log(2/delta)

    if method == "standard_gaussian":
        sample_size = int(np.ceil(8*c))
    elif method == "rademacher":
        sample_size = int(np.ceil(6*c))
    else:
        raise NotImplementedError

    return hutchinson_trace(A, sample_size=sample_size, method=method, block_size=block_size)



def hutch_plus_plus_trace(A, sample_size=30, method="rademacher"):
    """Computes the Hutch++ randomized estimator of tr(A). A must be SPSD. This is an improved estimator over
    the Hutchinson estimator. See [9].
    
    sample_size must be a multiple of 3.
    """

    # Get shape
    n = A.shape[0]

    valid_methods = ["standard_gaussian", "rademacher"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    assert sample_size % 3 == 0, "sample_size must be a multiple of 3."
    
    if method == "rademacher":
        S = np.random.choice([-1, 1], size=(n, int(sample_size/3)))
        G = np.random.choice([-1, 1], size=(n, int(sample_size/3)))
    elif method == "standard_gaussian":
        S = np.random.normal(size=(n, int(sample_size/3)))
        G = np.random.normal(size=(n, int(sample_size/3)))
    else:
        raise NotImplementedError

    # Do QR decomp
    Q, _ = scipy_qr(A @ S, mode="economic")

    # Compute approximate trace
    term1 = np.trace(Q.T @ ( A @ Q ) )
    tmp =  A @ ( G - ( Q @ ( Q.T @ G ) ) )
    tmp2 = G.T @ ( tmp - Q @ ( Q.T @ tmp ) )
    term2 = (3/sample_size)*np.trace(tmp2)
    trace_estimate = term1 + term2
    
    return trace_estimate



def hutch_plus_plus_epsilon_delta_trace(A, epsilon=0.05, delta=0.05, method="rademacher"):
    """Computes an (epsilon, delta)-estimator of trace(A) using the Hutch++ algorithm. A must be SPSD. This uses lower-bounds from the literature to pick a sample size 
    for the Hutch++ estimator \hat{tr}(A) such that | \hat{tr}(A) - tr(A) | < epsilon*tr(A) with probability greater than 1 - delta. See [9]."""
    
    valid_methods = ["standard_gaussian", "rademacher"]
    assert method in valid_methods, f"method must be one of {valid_methods}"

    sample_size = int( np.ceil( (np.sqrt(np.log(1/delta))/epsilon) + np.log(1/delta) ) )
    sample_size = int(3*np.ceil(sample_size/3))

    return hutch_plus_plus_trace(A, sample_size=sample_size, method=method)




