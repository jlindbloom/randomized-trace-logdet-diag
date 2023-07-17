import numpy as np
from scipy.sparse.linalg import LinearOperator

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator

from .cg import relative_resigual_cg


class AinvCGLinearOperator(LinearOperator):
    """Subclass of LinearOperator that represents A^{-1}, where A^{-1} x is computed approximately
      by the conjugate gradient method."""

    def __init__(self, A, cg_tol=1e-4, cg_maxits=1000, use_prev=True):

        # Bind
        self.A = A
        self.cg_tol = cg_tol
        self.cg_maxits = cg_maxits
        self.x0 = None
        self.use_prev = use_prev
        self.shape = self.A.shape
        self.dtype = self.A.dtype

        # Super
        super().__init__(self.dtype, self.shape)

    def _matvec(self, x):
        # Compute approximate sol
        approx_sol = relative_resigual_cg(self.A, x, eps=self.cg_tol, maxits=self.cg_maxits, x0=self.x0)
        approx_sol = approx_sol["x"]
        if self.use_prev: self.x0 = approx_sol

        return approx_sol
    
    def _matmat(self, B):
        output_shape = (self.shape[0], B.shape[1])
        result = np.zeros(output_shape)
        for j in range(B.shape[1]):
            result[:,j] = self._matvec(B[:,j])
        return result
    
    def _rmatmat(self, B):
        return self._matmat(B)
    
    def _rmatvec(self, x):
        return self._matvec(x)
    
    
    
    
if CUPY_INSTALLED:
    
    class AinvCGCuPyLinearOperator(CuPyLinearOperator):
        """Subclass of CuPyLinearOperator that represents A^{-1}, where A^{-1} x is computed approximately
          by the conjugate gradient method."""

        def __init__(self, A, cg_tol=1e-4, cg_maxits=1000, use_prev=True):

            # Bind
            self.A = A
            self.cg_tol = cg_tol
            self.cg_maxits = cg_maxits
            self.x0 = None
            self.use_prev = use_prev
            self.shape = self.A.shape
            self.dtype = self.A.dtype

            # Super
            super().__init__(self.dtype, self.shape)

        def _matvec(self, x):
            # Compute approximate sol
            approx_sol = relative_resigual_cg(self.A, x, eps=self.cg_tol, maxits=self.cg_maxits, x0=self.x0)
            approx_sol = approx_sol["x"]
            if self.use_prev: self.x0 = approx_sol

            return approx_sol

        def _matmat(self, B):
            output_shape = (self.shape[0], B.shape[1])
            result = cp.zeros(output_shape)
            for j in range(B.shape[1]):
                result[:,j] = self._matvec(B[:,j])
            return result

        def _rmatmat(self, B):
            return self._matmat(B)

        def _rmatvec(self, x):
            return self._matvec(x)

    
    
    
    
    