from pathlib import Path

import numpy as np
import pandas as pd

from spectraltools.io.parse_tsg import read_tsg_bip_pair
from spectraltools.extraction.extraction import extract_spectral_features

import matplotlib.pyplot as plt

def main():
    path = Path(r"C:\2022\minex-crc-op9\msdp01")
    files = ["287995_MSDP01_tsg.bip", "287995_MSDP01_tsg.tsg"]
    tsg_data = read_tsg_bip_pair(path/files[1], path/files[0], 'nir')
    feats = extract_spectral_features(tsg_data.spectra,tsg_data.ordinates, max_features=2, do_hull=True, ordinates_inspection_range=[2100.0, 2450.0], distance=4, main_guard=True, fit_type='cheb', resolution=2)
    bob=0
if __name__ == '__main__':
    main()