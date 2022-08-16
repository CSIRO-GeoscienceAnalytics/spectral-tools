""" _summary_

Returns:
    _description_
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from spectral.io import envi


@dataclass
class Spectra:
    """ _summary_
    """
    ordinates: NDArray
    spectra: NDArray
    header_info: dict

def read_envi(envi_datafile: Union[Path, str], envi_hdrfile: Union[Path, str]) -> Spectra:
    """read_envi _summary_

    Args:
        envi_datafile: _description_
        envi_hdrfile: _description_

    Returns:
        _description_
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
