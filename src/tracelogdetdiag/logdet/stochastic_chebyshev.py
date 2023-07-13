import numpy as np
from scipy.sparse.linalg import eigs as scipy_eigs

from .util import get_chebyshev_coeff


def logdet_stochastic_chebyshev_approx(C, sigma_max=None, sigma_min=None, sample_size=100, chebyshev_n=14):
    """Computes an approximation to logdet(C) for a SPSD matrix C, using the 
    stochastic Chebyshev approximation detailed in [7]. Eigenvalues of C are assumed to lie in
    the interval [sigma_min, sigma_max].

    Modified from author code here: https://alinlab.kaist.ac.kr/publications.html.
    """

    # Get dimension
    d = C.shape[0]
    
    if (sigma_max is None) and (sigma_min is None):

        # Get largest and smallest singular values
        sigma_max, _ = scipy_eigs(C, k=1, which="LM")
        sigma_min, _ = scipy_eigs(C, k=1, which="SM")
        sigma_max, sigma_min = np.real(sigma_max[0]), np.real(sigma_min[0])

    # Scaling
    a = sigma_min + sigma_max
    delta = sigma_min/a

    # Make B
    B = (1/a)*C
    logdet_estimate = 0.0

    # Funcs
    f = lambda x: np.log(1-x)
    g = lambda x: ((1-2*delta)/2)*x + 0.5
    ginv = lambda x: (2/(1-2*delta))*x
    h = lambda x: f(g(x))

    # Get Chebyshev coeffs
    chebyshev_coeffs = [ get_chebyshev_coeff(h, chebyshev_n, i) for i in range(0, chebyshev_n+1) ]

    # Random sampling
    for j in range(sample_size):
        
        # Draw random vector
        v = np.random.choice([-1, 1], size=d)
        u = chebyshev_coeffs[0]*v

        if chebyshev_n > 1:
            w0 = v
            w1 = B @ v
            w1 = ginv(w1)
            w1 = v/(1 - 2*delta) - w1
            u = chebyshev_coeffs[1]*w1 + chebyshev_coeffs[0]*w0

            for k in range(2, chebyshev_n+1):

                w2 = B @ w1
                w2 = ginv(w2)
                w2 = w1/(1 - 2*delta) - w2
                w2 = 2*w2 - w0
                u = chebyshev_coeffs[k]*w2 + u
                w0 = w1
                w1 = w2
        
        logdet_estimate += (np.dot(v, u))/sample_size

    logdet_estimate += d*np.log(a)

    return logdet_estimate



def logdet_stochastic_chebyshev_epsilon_delta_approx(C, epsilon=0.1, zeta=0.1, sample_size=None, details=False):
    """Computes an approximation to logdet(C) for a SPD matrix C, using the 
    stochastic Chebyshev approximation detailed in [7]. Returns an estimate
    \hat{logdet}(C) s.t. |logdet(C) - \hat{logdet}(C)| < epsilon*|logdet(C)| 
    with at least probaility 1-zeta. 

    If you override sample_size (which you might do since the bound is loose), you
    no longer have the same guarantee.

    Modified from author code here: https://alinlab.kaist.ac.kr/publications.html.
    """

    # Get largest and smallest singular values
    sigma_max, _ = scipy_eigs(C, k=1, which="LM")
    sigma_min, _ = scipy_eigs(C, k=1, which="SM")
    sigma_max, sigma_min = np.real(sigma_max[0]), np.real(sigma_min[0])
    kappa = sigma_max/sigma_min

    # Compute M and N
    M = (14/(epsilon**2))*((np.log(1 + (kappa**2)))**2)*np.log(2/zeta) # lower bound on sample size
    N_denom = np.log( ( np.sqrt(2*(kappa**2) + 1) + 1  ) / ( np.sqrt(2*(kappa**2) + 1) - 1 )  )
    N_num = np.log( (20/epsilon)*( np.sqrt( 2*(kappa**2) + 1 ) - 1 )*( (np.log(2 + 2*(kappa**2))) / (np.log(1 + (1/(kappa**2))))  )  )
    N = N_num/N_denom # chebyshev_n

    M = int(np.ceil(M))
    N = int(np.ceil(N))

    if details == True:
        print(f"Using {M} samples.")
        print(f"Using Chebyshev polynomials of order {N}.")

    return logdet_stochastic_chebyshev_approx(C, sigma_max, sigma_min, sample_size=M, chebyshev_n=N)




