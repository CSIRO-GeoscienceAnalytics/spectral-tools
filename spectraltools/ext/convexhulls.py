from numpy.typing import NDArray

from spectraltools.ext import chulls

def uc_hulls(ordinates: NDArray, data: NDArray, hull_type: int=0) -> NDArray:
    return chulls.get_absorption(ordinates, data, hull_type)

