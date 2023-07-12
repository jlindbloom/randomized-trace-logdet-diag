import numpy as np



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
        block_sum = np.sum( ( w.T.dot(A)*w.T).sum(axis=1) )
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
