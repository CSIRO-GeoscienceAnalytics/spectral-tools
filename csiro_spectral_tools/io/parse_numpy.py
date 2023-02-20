from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Spectra:
    """ 
    Data from calling read_numpy. Its a bit naff really as its already in a numpy format and
    all this will do is stick it into this class so its together. Cant really see this being of much use.
    """
    ordinates: NDArray
    spectra: NDArray

def read_numpy(numpy_ordinates: NDArray, numpy_spectra: NDArray) -> Spectra:
    """read_numpy 
    Take 2 numpy arrays of ordinates and spectral data and drop them into a class containing....
    wait for it, 2 numpy arrays :)
    This is a dumb one that really is not needed

    Args:
        numpy_ordinates (NDArray): ordinates of the spectral data [#bands]
        numpy_spectra (NDArray): the spectral data, 1D [#bands], 2D [N, #bands], 3D [N, M, #bands]

    Returns:
        Spectra: A class containing both things in one class. Wow!!
    """
    return Spectra(numpy_ordinates, numpy_spectra)
