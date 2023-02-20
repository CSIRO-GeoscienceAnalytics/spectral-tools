
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from spectral.io import envi


@dataclass
class Spectra:
    """ 
    Data returned from calling read_envi. Contains the spectral data, the ordinates and the header file information.
    """
    ordinates: NDArray
    spectra: NDArray
    header_info: dict

def read_envi(envi_datafile: Union[Path, str], envi_hdrfile: Union[Path, str]) -> Spectra:
    """
    Read an envi spectral file. An ENVI file has a binary file and an accompanying .hdr file

    Args:
        envi_datafile: pathlib Path or string literal of the envi binary files
        envi_hdrfile: pathlib Path or string literal of the header file for the above

    Returns:
        A class containing the ordinates, spectra and data contained in the header file
    """
    spectral_data: Union[envi.SpyFile, envi.SpectralLibrary]
    header_data: dict
    ordinates: NDArray
    spectral_data_out: NDArray

    if isinstance(envi_datafile, Path):
        envi_datafile = str(envi_datafile)

    if isinstance(envi_hdrfile, Path):
        envi_hdrfile = str(envi_hdrfile)

    spectral_data = envi.open(envi_hdrfile, image=envi_datafile)
    header_data = envi.read_envi_header(envi_hdrfile)
    ordinates = np.asarray(spectral_data.bands.centers)

    if isinstance(spectral_data, envi.SpectralLibrary):
        spectral_data_out = spectral_data.spectra

    if isinstance(spectral_data, envi.SpyFile):
        spectral_data_out = spectral_data[:, :, :]

    package = Spectra(ordinates, spectral_data_out, header_data)
    return package
