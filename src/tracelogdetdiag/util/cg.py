import numpy as np

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator
    
    

def relative_resigual_cg(A, b, x0=None, eps=1e-8, maxits=1000):
    """Applies the conjugate gradient method for the solution of A x = b 
    until || A x - b  || / || b || < eps.
    """
    
    # Figure out shape
    n = A.shape[0]
    
    # Handle CuPy
    if CUPY_INSTALLED:
        if isinstance(A, CuPyLinearOperator):
            xp = cp
        else:
            xp = np
    else:
        xp = np
    
    # b norm
    bnorm = xp.linalg.norm(b)
    
    # Initialization
    if x0 is None:
        x = xp.ones(n)
    else:
        x = x0
    
    converged = False
    r = b - A.matvec(x)
    d = r.copy()
    
    its = 0
    for j in range(maxits):
        
        alpha = (r.T @ r)/(d.T @ A.matvec(d) )
        x = x + alpha*d
        rnew = r - alpha * A.matvec( d )
        beta = (rnew.T @ rnew)/(r.T @ r)
        d = rnew + beta*d
        r = rnew
        
        its += 1
        residual_norm = xp.linalg.norm( b - A.matvec( x ) )
        rel_residual_norm = residual_norm/bnorm
        if rel_residual_norm < eps: 
            converged = True
            break

    assert converged, "CG didn't converge in less than maxits iterations!"
        
    data = {
        "x": x,
        "iterations": its,
    }
    
    return data

