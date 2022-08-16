""" _summary_

Returns:
    _description_

Yields:
    _description_
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Spectra:
    """ _summary_
    """
    ordinates: NDArray
    spectra: NDArray
    sample_names: list[str]

def _get_csv_data(csv_file: Union[Path, str], csv_order: int = 1) -> tuple[NDArray, NDArray, list[str]]:
    """_get_csv_data _summary_

    Args:
        csv_file: _description_
        csv_order: _description_. Defaults to 1.

    Returns:
        _description_
    """
    names: list[str]
    spectra: NDArray
    ordinates: NDArray

    if isinstance(csv_file, str):
        csv_file = Path(csv_file)
    if csv_order == 0:
        df_csv_spectra = pd.read_csv(csv_file, index_col=0)
        names = df_csv_spectra.index.values.tolist()
        spectra = df_csv_spectra.values.astype(float)
        ordinates = df_csv_spectra.columns.values.astype(float)
    else:
        df_csv_spectra = pd.read_csv(csv_file)
        names = df_csv_spectra.columns.values[1:].tolist()
        spectra = np.transpose(df_csv_spectra.values[:, 1:]).astype(float)
        ordinates = df_csv_spectra.iloc[1:, 0].values.astype(float)

    return ordinates, spectra, names

    # def datagenerator(self):
    #     """
    #     an iterator to return a single spectrum at a time from the dataset
    #     Yields:
    #         a single spectrum from the dataset

    #     """
    #     for spectrum in self.spectra:
    #         yield spectrum


def read_csv(csv_file: Union[Path, str], csv_order: int = 1) -> Spectra:
    """read_csv _summary_

    Args:
        csv_file: _description_
        csv_order: _description_. Defaults to 1.

    Returns:
        _description_
    """
    csv_wavelengths, csv_spectra, sample_names = _get_csv_data(csv_file, csv_order)
    package = Spectra(csv_wavelengths, csv_spectra, sample_names)
    return package

