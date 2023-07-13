import numpy as np



def evaluate_ith_chebyshev_polynomial(xs, i):
    """Evaluates the ith Chebyshev polynomial at imputs xs."""
    tprev = np.ones_like(xs)
    tnext = xs.copy()
    if i == 0:
        return tprev
    elif i == 1:
        return tnext
    else:
        k = 1
        while k < i:
            tnew = 2*xs*tnext - tprev
            tprev = tnext
            tnext = tnew
            k += 1
        return tnew
    


def get_chebyshev_coeff(f, n, i):
    """Computes the ith Chebyshev coefficient in the expansion
            f(x) \approx \sum_{j=0}^n c_j T_j(x).
    Here f:[-1,1] \to \mathbb{R}. 
    """

    # Get nodes
    ks = np.arange(n+1)
    xks = np.cos(  np.pi*( ks + 0.5 )/(n+1)  )

    if i == 0:
        chebyshev_coeff = (1/(n+1))*(f(xks)*1).sum()
    else:
        chebyshev_coeff = (2/(n+1))*(f(xks)*evaluate_ith_chebyshev_polynomial(xks, i) ).sum()
        

    return chebyshev_coeff















