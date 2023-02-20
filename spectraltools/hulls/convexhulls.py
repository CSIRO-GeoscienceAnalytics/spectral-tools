"""
Takes a spectral array and its ordinates and calculates a hull solution.
This is a wrapper to the extension chulls in spectraltools.ext.chulls

Returns:
    NDArray: hull quotient or hull removed or hull
"""
from numpy.typing import NDArray

from spectraltools.ext import chulls


def uc_hulls(ordinates: NDArray, data: NDArray, hull_type: int = 0) -> NDArray:
    """uc_hulls Calculate either a hull quotient, hull removal or hull

    Args:
        ordinates (NDArray): ordinates of the spectra
        data (NDArray): 1D, 2D or 3D array of spectral data. Last dimension is the spectra
        hull_type (int, optional): 0 = hull quotient, 1 = hull removal, 2 = hull. Defaults to 0.

    Returns:
        NDArray: hull quotient or hull removed or hull
    """
    return chulls.get_absorption(ordinates, data, hull_type)
