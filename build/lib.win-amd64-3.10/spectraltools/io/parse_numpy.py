""" _summary_

Returns:
    _description_
"""
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Spectra:
    """ _summary_
    """
    ordinates: NDArray
    spectra: NDArray

def read_numpy(numpy_ordinates: NDArray, numpy_spectra: NDArray) -> Spectra:
    """read_numpy _summary_

    Args:
        numpy_ordinates: _description_
        numpy_spectra: _description_

    Returns:
        _description_
    """

    return Spectra(numpy_ordinates, numpy_spectra)
