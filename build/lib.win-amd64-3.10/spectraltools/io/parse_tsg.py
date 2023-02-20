""" The parse_tsg Module
"""

import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
import pandas as pd
from numpy.core._exceptions import _ArrayMemoryError
from numpy.typing import NDArray
from PIL import Image


class CrasHeader(NamedTuple):
    """CrasHeader _summary_

    Args:
        NamedTuple: _description_
    """
    id: str  # starts with "CoreLog Linescan ".   If it starts with "CoreLog Linescan 1." then it supports compression (otherwise ignore ctype).
    ns: int  # image width in pixels
    nl: int  # image height in lines
    nb: int  # number of bands (1 or 3  but always 3 for HyLogger 1 / 2 / 3)
    org: int  # interleave (1=BIL  2=BIP  and compressed rasters are always BIP while uncompressed ones are always BIL)
    dtype: int  # datatype (unused  always byte)
    specny: int  # number of linescan lines per dataset sample
    specnx: int  # unused
    specpx: int  # unused  intended to be the linescan column that relates to the across-scan position of the (1D) spectral dataset
    ctype: int  # compression type (0=uncompressed  1=jpeg chunks)
    chunksize: int  # number of image lines per chunk for jpeg-compressed rasters
    nchunks: int  # number of compressed image chunks (jpeg compression)
    csize32_obs: int  # size in bytes of comressed image data (OBSOLETE - not used anywhere any more.   However it will be set in old linescan rasters so I cant easily recycle it.   Also  there are some compressed rasters out there that are >4GB in size)
    ntrays: int  # number of trays (number of tray-table records after the image data)
    nsections: int  # number of sections (number of section-table records after the image data)
    finerep: int  # chip-mode datasets - number of spectral measurements per chip bucket (and theres one image frame per bucket)
    jpqual: int  # jpeg quality factor  0..100 (jpeg compression)


class TrayInfo(NamedTuple):
    """TrayInfo _summary_

    Args:
        NamedTuple: _description_
    """
    utlengthmm: float  # "untrimmed" length of tray imagery in mm
    baseheightmm: float  # height of bottom of tray above table
    coreheightmm: float  # height of (top of) core above ..something or other (I don't actually use it for the linescan raster)
    nsections: int  # number of core sections
    nlines: int  # number of image lines in this tray


class SectionInfo(NamedTuple):
    utlengthmm: float  # untrimmed length of imagery in mm (could be less than the tray's)
    startmm: float  # start position (along scan) in mm
    endmm: float  # end position in mm
    trimwidthmm: float  # active (section-overlap-corrected) image width in mm
    startcol: int  # number of image lines in this tray
    endcol: int  # end pixel across for active (section-overlap-corrected) imagery
    nlines: int  # number of image lines in this section


@dataclass
class Cras:
    image: NDArray
    tray: "list[TrayInfo]"
    section: "list[SectionInfo]"


@dataclass
class Spectra:
    spectrum_name: str
    spectra: NDArray
    ordinates: NDArray
    #bandheaders: list[dict[str, str]] #"list[str]"
    sampleheaders: pd.DataFrame
    #classes: dict[str, tuple[dict, dict]] #"list[[dict[str, Any]]]"
    bandheader_dataframe: pd.DataFrame

@dataclass
class TSG:
    nir: Spectra
    tir: Spectra
    cras: Cras
    lidar: Union[NDArray,None]


class FilePairs:
    """Class for keeping track of an item in inventory."""

    nir_tsg: Union[Path, None] = None
    nir_bip: Union[Path, None] = None
    tir_tsg: Union[Path, None] = None
    tir_bip: Union[Path, None] = None
    lidar: Union[Path, None] = None
    cras: Union[Path, None] = None

    def _get_bip_tsg_pair(self, spectrum: str):
        tsgfile: Union[Path, None] = getattr(self, f"{spectrum}_tsg")
        bipfile: Union[Path, None] = getattr(self, f"{spectrum}_bip")

        has_tsg: bool = isinstance(tsgfile, Path)
        has_bip: bool = isinstance(bipfile, Path)
        names_match: bool
        if has_tsg and has_bip:
            names_match = bipfile.stem == tsgfile.stem
        else:
            names_match = False

        if has_bip and has_tsg and names_match:
            pairs = (tsgfile, bipfile)
        else:
            pairs = None
        return pairs

    def _get_lidar(self) -> Union[Path, None]:
        has_lidar: bool = ("lidar" in self.__dict__.keys()) and (
            isinstance(self.lidar, Path)
        )
        if has_lidar:
            pairs = self.lidar
        else:
            pairs = None
        return pairs

    def _get_cras(self) -> Union[Path, None]:
        has_cras: bool = isinstance(self.cras, Path)
        if has_cras:
            pairs = self.cras
        else:
            pairs = None
        return pairs

    def valid_nir(self) -> bool:
        result = self._get_bip_tsg_pair("nir")
        if result is None:
            valid = False
        else:
            valid = True
        return valid

    def valid_tir(self) -> bool:
        result = self._get_bip_tsg_pair("tir")
        if result is None:
            valid = False
        else:
            valid = True
        return valid

    def valid_lidar(self) -> bool:
        result = self._get_lidar()
        if result is None:
            valid = False
        else:
            valid = True
        return valid

    def valid_cras(self) -> bool:
        result = self._get_cras()
        if result is None:
            valid = False
        else:
            valid = True
        return valid

def read_cras(filename: Union[str, Path]) -> Cras:
    """ Read a cras file

    Args:
        filename: filename to read

    Returns:
        A cras object
    """
    section_info_format: str = "4f3i"
    tray_info_format: str = "3f2i"
    head_format: str = "20s2I8h4I2h"

    with open(filename, "rb") as file:
        # using memory mapping

        # file = mmap.mmap(fopen.fileno(), 0)

        # read the 64 byte header from the .cras file
        bytes_in = file.read(64)
        # create the header information
        header = CrasHeader(*struct.unpack(head_format, bytes_in))

        # Create the chunk_offset_array
        # which determines which point of the file to enter to read the .jpg image

        file.seek(64)
        b = file.read(4 * (header.nchunks + 1))
        chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)
        # we currently are reading the entire cras.bip file
        # which can cause issues due to memory allocation  well handle that case
        # with some error handling here where we quit while we are ahead
        try:
            cras = np.zeros((header.nl, header.ns, header.nb), dtype=np.uint8)
            array_ok = True
        except _ArrayMemoryError:
            array_ok = False
            cras = np.zeros(1, dtype=np.uint8)
        # if the array fits into memory then proceed to decode the .jpgs
        # using the chunk_offset_array to correctly index to the right location
        # when you might like to have images saved if the file is too big to fit
        # into ram

        if array_ok:
            curpos: int = 0
            nr: int
            for i in range(header.nchunks):
                total_offset = chunk_offset_array[i] + 4 * (header.nchunks + 1) + 64
                chunksize_in_bytes = chunk_offset_array[i + 1] - chunk_offset_array[i]
                file.seek(total_offset)
                chunk = file.read(chunksize_in_bytes)
                img = Image.open(io.BytesIO(chunk))
                # reverse the channels
                # and flip the image upsidedown
                np_image = np.flipud(np.flip(np.array(img), -1))
                nr = np_image.shape[0]
                cras[curpos : (curpos + nr), :, :] = np_image
                curpos = curpos + nr

        # the tray info section if it exists should start after the last image
        # the section info and if there is a tray info section then it should be after the tray info section
        info_table_start = (
            64
            + (header.nchunks + 1) * 4
            + chunk_offset_array[header.nchunks]
            - chunk_offset_array[0]
        )
        file.seek(info_table_start)

        tray: list[TrayInfo] = []
        for i in range(header.ntrays):
            bytes_in = file.read(20)
            tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes_in)))

        section: list[SectionInfo] = []
        for i in range(header.nsections):
            bytes_in = file.read(28)
            section.append(SectionInfo(*struct.unpack(section_info_format, bytes_in)))

    output = Cras(cras, tray, section)
    return output


def _read_tsg_file(filename: Union[str, Path]) -> "list[str]":
    """ Reads the files with the .tsg extension which are almost a toml file
    but not quite so the standard parser doesn't work

    Quite simply this function reads the file and strips the newlines at the end
    to simplify processing later on

    """
    lines: list[str] = []
    tmp_line: str
    with open(filename) as file:
        for line in file:
            tmp_line = line.rstrip()
            lines.append(tmp_line)
    return lines


def _find_header_sections(tsg_str: "list[str]"):
    """ Finds the header sections of the .tsg file
    header sections are defined as strings between square brackets
    """
    re_strip: re.Pattern = re.compile("^\\[[a-zA-Z0-9 ]+\\]")
    positions: "list[int]" = []
    for i, s in enumerate(tsg_str):
        if len(re_strip.findall(s)) > 0:
            positions.append(i)
    # this final position if the end of file
    # it is used to ensure that the final loop when pairing
    # iterates over the last item
    positions.append(len(tsg_str))
    n_headers: int = len(positions)
    sections: "dict[str, tuple[int,int]]" = {}
    tmp_section: "tuple[int,int]"
    tmp_name: str
    for i in range(n_headers - 1):
        tmp_section = (positions[i] + 1, positions[i + 1] - 1)
        tmp_name = tsg_str[positions[i]].strip("[]")
        sections.update({tmp_name: tmp_section})
    return sections


def _parse_section(
    section_list: "list[str]", key_split: str = ":"
) -> "list[dict[str, str]]":
    """
    machine id = 0
    ag.c3 = 0.000000
    8:Width2
    """
    final: list[dict[str, str]] = []
    kvp: dict[str, str]
    for i in section_list:
        kvp = _parse_kvp(i, key_split)
        final.append(kvp)

    return final


def _parse_sample_header(
    section_list: "list[str]", key_split: str = ":"
) -> "list[dict[str, str]]":
    """
    '0:ETG0187_0001_1  T=0001 L=1 P=1 D=1.000005 X=4.000000 H=ETG0187'
    '1:ETG0187_0001_2  T=0001 L=1 P=2 D=1.000006 X=12.000000 H=ETG0187'
    '2:ETG0187_0001_3  T=0001 L=1 P=3 D=1.000006 X=20.000000 H=ETG0187'
    """
    final: list[dict[str, str]] = []
    key_0: str
    tmp_sample: dict[str, str]

    for i in section_list:
        kk = _parse_kvp(i, key_split)
        k0 = list(kk.keys())
        key_0: str = k0[0]
        tmp_sample = {}
        tmp_sample.update({"sample": key_0})
        for j in kk[key_0].split():
            tmp_keys = _parse_kvp(j)
            if not tmp_keys is None:
                tmp_sample.update(tmp_keys)
        final.append(tmp_sample)

    return final

def _parse_class_section(section_list: list[str]) -> tuple[dict, dict]: #"tuple[dict[str, str]]":
    """
    name = S_jCLST_707 Groups
    max = 2
    colours = 15126526 30960
    0:SILICA
    1:K-FELDSPAR
    """
    class_names:dict[str,str] = {}
    class_info:dict[str,str]  = {}
    for i in section_list:
        # lines of the section containing = form the class name
        if i.find('=')>=0:
            tmp_name = _parse_kvp(i,'=')
            class_names.update(tmp_name )
        elif i.find(':')>=0:
            tmp_info = _parse_kvp(i, ':')
            class_info.update(tmp_info)

    return class_names, class_info

def _parse_wavelength_specs(line: str) -> dict[str, Union[float,str]]:

    split_wavelength: list[str] = line.split()
    wavelength = {
        "start": float(split_wavelength[0]),
        "end": float(split_wavelength[1]),
        "unit": split_wavelength[-1],
    }
    return wavelength


def _parse_kvp(line: str, split: str = "=") -> "dict[str, str]":
    """ Parses strings into Key value pairs
    control over the split value is to manage the different seperators used
    in different sections of the file

    Args:
        line: the current line to parse
    Returns:
        a dictionary with key and value
    Examples:
        >>> line = 'name=ben'
        >>> parse_kvp(line)
        >>> {'name':'ben'}

    """
    if line.find(split) >= 0:
        split_line = line.split(split)
        key = split_line[0].strip()
        value = split_line[1].strip()
        kvp = {key: value}
    else:
        kvp = {}
    return kvp


def _read_bip(filename: Union[str, Path], coordinates: "dict[str, str]") -> NDArray[np.float32]:
    """ Reads the .bip file as a 1d array then reshapes it according to the dimensions
    as supplied in the coordinates dict

    Args:
        filename: location of the .bip file
        coordinates: dimension of the .bip file
    Returns:
        a 3d numpy array the first dimension corresponds to the spectra and mask
        the second the samples
        the third the bands
    Examples:
    """
    # load array in 1d
    tmp_array: NDArray[np.float32] = np.fromfile(filename, dtype=np.float32)

    # extract information on array shape
    n_bands: int = int(coordinates["lastband"])
    n_samples: int = int(coordinates["lastsample"])
    # reshape array
    spectrum = np.reshape(tmp_array, (2, n_samples, n_bands))
    return spectrum


def _calculate_wavelengths(
    wavelength_specs: "dict[str,float]", coordinates: "dict[str, str]"
) -> NDArray:
    wavelength_range: float = wavelength_specs["end"] - wavelength_specs["start"]
    resolution: float = wavelength_range / (int(coordinates["lastband"]) - 1)

    return np.arange(
        wavelength_specs["start"], wavelength_specs["end"] + resolution, resolution
    )


def read_hires_dat(filename: Union[str, Path]) -> NDArray:
    """ Read the *hires.dat* file which contains the lidar scan
    of the material

    Args:
        filename: location of the .dat file
    Returns:
        np.ndarray representing the
    Examples:
    """
    # the hires .dat file is f32 and the actual data starts at pos 640
    # the rest is probably information pertaining to the instrument itself
    lidar = np.fromfile(filename, dtype=np.float32, offset=640)
    return lidar


def _parse_tsg(
    fstr: "list[str]", headers: "dict[str, tuple[int,int]]"
) -> "dict[str, Any]":
    d_info: dict[str, Any] = {}
    tmp_header: "list[dict[str, str]]" = []
    start: int
    end: int

    for k in headers.keys():
        start = headers[k][0]
        end = headers[k][1]
        if k == "sample headers":
            tmp_header = _parse_sample_header(fstr[start:end], ":")
            d_info.update({k: pd.DataFrame(tmp_header)})
        elif k == "wavelength specs":
            tmp_wave = _parse_wavelength_specs(fstr[start:end][0])
            d_info.update({k: tmp_wave})
        elif k == "band headers":
            tmp_header = _parse_section(fstr[start:end], ":")
            d_info.update({k: tmp_header})
        elif k.find("class") == 0:
            tmp_header,tmp_info = _parse_class_section(fstr[start:end])
            tmp_class = {k: (tmp_header,tmp_info)}
            if "class" in d_info.keys():
                d_info['class'].update(tmp_class)
            else:
                d_info.update({"class": tmp_class})

        else:
            tmp_out: dict[str, str] = {}
            for i in fstr[start:end]:
                tmp = _parse_kvp(i)
                if not tmp is None:
                    tmp_out.update(tmp)
            d_info.update({k: tmp_out})

    return d_info

def _create_bandheader_df(info: dict[str, Any], spectra: NDArray[np.float32]) -> pd.DataFrame:
    """_create_bandheader_df _summary_

    Args:
        info: _description_
        spectra: _description_

    Returns:
        _description_
    """
    single_info_dict: dict
    bandheader_indices: list = []
    bandheader_names: list = []
    bandheader_data: pd.DataFrame

    # merge the individual dictionaries into a singular dictionary for ease
    single_info_dict = dict()
    for _dict in info["band headers"]:
        single_info_dict |= _dict
    # extract the indices we need to extract the data
    bandheader_indices = [int(i) for i in list(single_info_dict.keys())]
    # get the bandheader names
    bandheader_names = [single_info_dict.get(val).split(';')[0] for val in single_info_dict.keys()]
    # create a dataframe so its all in one place and easily accesible :)
    bandheader_data = pd.DataFrame(spectra[1, :, bandheader_indices].T, columns=bandheader_names)
    # find entries that have a class mapping
    for i, key in enumerate(bandheader_indices):
        value = single_info_dict[str(key)]
        value_bits = value.split(";")
        if len(value_bits) > 1:
            if value_bits[3] == '2':
                if int(value_bits[4]) >= 0:
                    class_info = info["class"]["class " + value_bits[4]][1]
                    bandheader_data[value_bits[0]] = bandheader_data[value_bits[0]].astype(int).astype(str).map(class_info)

    return bandheader_data

def read_tsg_bip_pair(tsg_file: Union[Path, str], bip_file: Union[Path, str], spectrum: str) -> Spectra:
    fstr = _read_tsg_file(tsg_file)
    headers = _find_header_sections(fstr)
    info = _parse_tsg(fstr, headers)
    spectra = _read_bip(bip_file, info["coordinates"])
    bandheader_df = _create_bandheader_df(info, spectra)
    wavelength = _calculate_wavelengths(info["wavelength specs"], info["coordinates"])
    package = Spectra(
        spectrum, spectra[0,:,:], wavelength, info["sample headers"], bandheader_df
    )

    return package


def read_package(foldername: Union[str, Path], read_cras_file: bool = False) -> TSG:
    # convert string to Path because we are wanting to use Pathlib objects to manage the folder structure
    if isinstance(foldername, str):
        foldername = Path(foldername)

    if not foldername.exists():
        raise FileNotFoundError("The directory does not exist.")

    # we are parsing the folder structure here and checking that
    # pairs of files exist in this case we are making sure
    # that there are .tsg files with corresponding .bip files
    # we will parse the lidar height data because we can

    # process here is to map the files that we need together
    # tir and nir files
    #
    # deal the files to the type

    f: Path
    file_pairs = FilePairs()
    files = foldername.glob("*.*")
    for f in files:
        if f.name.endswith("tsg.tsg"):
            setattr(file_pairs, "nir_tsg", f)

        elif f.name.endswith("tsg.bip"):
            setattr(file_pairs, "nir_bip", f)

        elif f.name.endswith("tsg_tir.tsg"):
            setattr(file_pairs, "tir_tsg", f)

        elif f.name.endswith("tsg_tir.bip"):
            setattr(file_pairs, "tir_bip", f)

        elif f.name.endswith("tsg_cras.bip"):
            setattr(file_pairs, "cras", f)

        elif f.name.endswith("tsg_hires.dat"):
            setattr(file_pairs, "lidar", f)
        else:
            pass

    # once we have paired the .tsg and .bip files run the reader
    # for the nir/swir and then tir
    # read nir/swir
    nir: Spectra
    tir: Spectra
    lidar: Union[NDArray, None]
    cras: Cras

    if file_pairs.valid_nir():
        nir = read_tsg_bip_pair(file_pairs.nir_tsg, file_pairs.nir_bip, "nir")
    else:
        nir = Spectra

    if file_pairs.valid_tir():
        tir = read_tsg_bip_pair(file_pairs.tir_tsg, file_pairs.tir_bip, "tir")
    else:
        tir = Spectra
    if file_pairs.valid_lidar():
        lidar = read_hires_dat(file_pairs.lidar)
    else:
        lidar = None
    if file_pairs.valid_cras() and read_cras_file:
        cras = read_cras(file_pairs.cras)
    else:
        cras = Cras

    return TSG(nir, tir, cras, lidar)
