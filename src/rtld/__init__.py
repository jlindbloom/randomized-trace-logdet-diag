# Check if cupy is installed or not
CUPY_INSTALLED = False
try:
    import cupy
    CUPY_INSTALLED = True
except:
    pass