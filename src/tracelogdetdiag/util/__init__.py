from .cg import relative_resigual_cg
from .AinvCGLinearOperator import AinvCGLinearOperator

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    from .AinvCGLinearOperator import AinvCGCuPyLinearOperator
