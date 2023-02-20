from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Spectra:
    ordinates: NDArray
    spectra: NDArray
    sample_names: List[str]


def _get_csv_data(csv_file: Union[Path, str], csv_order: int = 1) -> Tuple[NDArray, NDArray, List[str]]:
    names: List[str]
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


def read_csv(csv_file: Union[Path, str], csv_order: int = 1) -> Spectra:
    """read_csv
    Gather up wavelengths, spectral data and spectra names from a csv file. This is just a small wrapper
    around a pandas read_csv

    Args:
        csv_file (Union[Path, str]):pathlib Path to csv file or string literal to file
        csv_order (int, optional): 1 = wavelength first column, 0 = wavelength first row. Defaults to 1.

    Returns:
        Spectra: Spectra class
    """
    # """read_csv Read spectral data from a CSV file

    # Args:
    #     csv_file (Union[Path, str]): The path to the file
    #     csv_order (int, optional): CSV order. Defaults to 1.

    # Returns:
    #     Spectra: The spectra, ordinates and spectral sample names
    # """
    csv_wavelengths, csv_spectra, sample_names = _get_csv_data(csv_file, csv_order)
    package = Spectra(csv_wavelengths, csv_spectra, sample_names)
    return package
