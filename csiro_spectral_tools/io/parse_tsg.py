import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
from numpy.core._exceptions import _ArrayMemoryError
from numpy.typing import NDArray
from simplejpeg import decode_jpeg, encode_jpeg


class ClassHeaders(NamedTuple):
    class_number: int
    name: str
    max: int
    classes: "Dict[int, str]"

    def map_ints(self, index: NDArray) -> "List[str]":
        outindex: List[str] = []
        for i in index:
            if i >= 0:
                outindex.append(self.classes[i])
            else:
                outindex.append("")
        return outindex


class CrasHeader(NamedTuple):
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


class BandHeaders(NamedTuple):
    band: int
    name: str
    class_number: int
    flag: int  # I'm not sure what this does but 2 indicates that it is a mappable class


@dataclass
class Cras:
    image: NDArray
    tray: "List[TrayInfo]"
    section: "List[SectionInfo]"


@dataclass
class Spectra:
    spectrum_name: str
    spectra: NDArray
    wavelength: NDArray
    sampleheaders: pd.DataFrame
    classes: "List[Dict[str, Any]]"
    bandheaders: "List[BandHeaders]"
    scalars: pd.DataFrame


@dataclass
class TSG:
    nir: Spectra
    tir: Spectra
    cras: Cras
    lidar: Union[NDArray, None]

    def __repr__(self) -> str:
        tsg_info: str = "This is a TSG file"
        return tsg_info


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
        has_lidar: bool = ("lidar" in self.__dict__.keys()) and (isinstance(self.lidar, Path))
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
    """Read a cras file

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
        bytes = file.read(64)
        # create the header information
        header = CrasHeader(*struct.unpack(head_format, bytes))

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
        # TODO: it might be worthwhile to modify this code to manage the case
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
                img = decode_jpeg(chunk, colorspace="BGR")
                # reverse the channels
                # and flip the image upsidedown
                np_image = np.flipud(img)
                nr = np_image.shape[0]
                cras[curpos : (curpos + nr), :, :] = np_image
                curpos = curpos + nr

        # the tray info section if it exists should start after the last image
        # the section info and if there is a tray info section then it should be after the tray info section
        info_table_start = (
            64 + (header.nchunks + 1) * 4 + chunk_offset_array[header.nchunks] - chunk_offset_array[0]
        )
        file.seek(info_table_start)

        tray: List[TrayInfo] = []
        for i in range(header.ntrays):
            bytes = file.read(20)
            tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

        section: List[SectionInfo] = []
        for i in range(header.nsections):
            bytes = file.read(28)
            section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

    output = Cras(cras, tray, section)
    return output


def extract_chips(filename: Union[str, Path], outfolder: Union[str, Path], spectra: Spectra):
    if isinstance(outfolder, str):
        outfolder = Path(outfolder)

    if not outfolder.exists():
        outfolder.mkdir()

    section_info_format: str = "4f3i"
    tray_info_format: str = "3f2i"
    head_format: str = "20s2I8h4I2h"
    file = open(filename, "rb")
    # read the 64 byte header from the .cras file
    bytes = file.read(64)
    # create the header information
    header = CrasHeader(*struct.unpack(head_format, bytes))

    # Create the chunk_offset_array
    # which determines which point of the file to enter to read the .jpg image

    file.seek(64)
    b = file.read(4 * (header.nchunks + 1))
    chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)

    # check for the existance of sections or  trays
    # if they exist we are going to skip ahead and import them first
    # as we are going to use them to section the images to a per spectrum basis
    # so the cras file uses compressed jpg chunks of approximately fixed dimension
    # so we need to use the section and tray information to calculate the correct
    # image size that matches the spectra
    # we will set up an array that we will use to accumulate the images into
    # then as each image is accumulated we dump it to disk and call name it the sample name
    # it is likely that we need two accumulation arrays the first as a bin to hold the images
    # as they are read to disk and the second to hold the image that we are going to export
    if header.nsections > 0 or header.ntrays > 0:
        # the tray info section if it exists should start after the last image
        # the section info and if there is a tray info section then it should be after the tray info section
        info_table_start = (
            64 + (header.nchunks + 1) * 4 + chunk_offset_array[header.nchunks] - chunk_offset_array[0]
        )
        file.seek(info_table_start)

        tray: "List[TrayInfo]" = []
        for i in range(header.ntrays):
            bytes = file.read(20)
            tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

        section: "List[SectionInfo]" = []
        for i in range(header.nsections):
            bytes = file.read(28)
            section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

    # it seems to be best to allocate memory for each of the sections if there are multiple sections we
    # empty the array and create a new one of the correct dimension
    # on third thoughts we will precalculate which chunks are going to which section because we know that
    # then loop over sets of chunks dumping to disk incrementally.
    # at this stage I'm not sure it will work on drill core
    # no the header contains the chunk dimensions
    # loop over the section
    # it seems that you need to have the sample header information from the
    # nir/tir spectra we use nir because it should always be there
    # once we have that information we are going to caculate the number of pixels required
    # in the y direction that represent a single spectrum and the option will also be to dump
    # all the spectra to disk named as H_SAMPLE in a subfolder which will take an impressive amount of space
    # but such are the vagaries of ML
    # I totally assume that these headers always exist in the scalars

    section_array = spectra.sampleheaders["L"].astype(int).values - 1  # subtract 1 so that we are 0 indexed
    sample_length = spectra.scalars["SecDist (mm)"].diff()
    # this is only na for the first sample
    idx_sample_na = (sample_length.isna()) | (sample_length < 0)
    sample_length[idx_sample_na] = spectra.scalars["SecDist (mm)"][idx_sample_na]
    # pd is slow for lots of accesses
    sample_array: NDArray = sample_length.values
    curchunk: int = 0
    cursample: int = 0
    processed_lines: int = 0
    yres: float
    working: NDArray[np.uint8]
    curpos: int = 0
    nr: int
    pos_fill: NDArray[np.int32]
    idx_bin_fill: NDArray[np.bool8]
    leading_bin: NDArray[np.uint8] = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
    total_offset: int
    chunksize_in_bytes: int
    np_image: NDArray[np.uint8]
    end_pos: int
    nextra: int
    end_np: int
    idx_section: NDArray[np.bool8]
    cut_array: NDArray[np.int32]
    n_cuts: int
    tmp_file: str
    for i, sec in enumerate(section):
        # pixel resolution
        yres = sec.utlengthmm / sec.nlines
        # allocate the section
        working = np.zeros((sec.nlines, header.ns, header.nb), dtype="uint8")
        curpos = 0
        # if we've dropped any information into the leading bin
        # dump it out into for into the working array
        if np.any(leading_bin):
            idx_bin_fill = np.all(np.any(leading_bin, 1), 1)
            pos_fill = np.where(idx_bin_fill)[0]
            working[pos_fill] = leading_bin[pos_fill]
            curpos = pos_fill[-1] + 1
            # empty the leading bin
            leading_bin = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
        # you need to monitor the processed lines to maintain this loop
        while (curchunk * header.chunksize - processed_lines) < sec.nlines:
            total_offset = chunk_offset_array[curchunk] + 4 * (header.nchunks + 1) + 64
            chunksize_in_bytes = chunk_offset_array[curchunk + 1] - chunk_offset_array[curchunk]
            file.seek(total_offset)
            chunk = file.read(chunksize_in_bytes)
            np_image = decode_jpeg(chunk, colorspace="BGR")
            np_image = np.flipud(np_image)
            nr = np_image.shape[0]
            end_pos = curpos + nr

            if end_pos <= sec.nlines:
                working[curpos:end_pos, :, :] = np_image
            elif end_pos > sec.nlines:
                nextra = end_pos - sec.nlines
                end_pos = sec.nlines
                end_np = nr - nextra
                working[curpos:end_pos, :, :] = np_image[0:end_np]
                # put the remaining information into leading bin
                leading_bin[0:nextra, :, :] = np_image[end_np:nr]

            curpos = curpos + nr
            # increment the chunk
            curchunk += 1

        # book keeping the processed lines
        processed_lines += sec.nlines

        idx_section = section_array == i
        im_cuts = np.floor(sample_array[idx_section] / yres).astype(int)
        cut_array = np.concatenate([[0], np.cumsum(im_cuts).ravel()])

        n_cuts = len(cut_array)
        for j in range(n_cuts - 1):
            current_image = working[cut_array[j] : cut_array[j + 1]]
            tmp_file = "{}.jpg".format(cursample)
            outfile = outfolder.joinpath(tmp_file)
            outjpg = encode_jpeg(current_image)
            with open(outfile, "wb") as tmpf:
                tmpf.write(outjpg)
            cursample += 1


def _read_tsg_file(filename: Union[str, Path]) -> "List[str]":
    """Reads the files with the .tsg extension which are almost a toml file
    but not quite so the standard parser doesn't work

    Quite simply this function reads the file and strips the newlines at the end
    to simplify processing later on

    """
    lines: List[str] = []
    tmp_line: str
    with open(filename, encoding="cp1252") as file:
        for line in file:
            tmp_line = line.rstrip()
            lines.append(tmp_line)
    return lines


def _find_header_sections(tsg_str: "List[str]"):
    """Finds the header sections of the .tsg file
    header sections are defined as strings between square brackets
    """
    re_strip: re.Pattern = re.compile("^\\[[a-zA-Z0-9 ]+\\]")
    positions: "List[int]" = []
    for i, s in enumerate(tsg_str):
        if len(re_strip.findall(s)) > 0:
            positions.append(i)
    # this final position if the end of file
    # it is used to ensure that the final loop when pairing
    # iterates over the last item
    positions.append(len(tsg_str))
    n_headers: int = len(positions)
    sections: "Dict[str, Tuple[int,int]]" = {}
    tmp_section: "Tuple[int,int]"
    tmp_name: str
    for i in range(n_headers - 1):
        tmp_section = (positions[i] + 1, positions[i + 1] - 1)
        tmp_name = tsg_str[positions[i]].strip("[]")
        sections.update({tmp_name: tmp_section})
    return sections


def _parse_section(section_list: "List[str]", key_split: str = ":") -> "List[Dict[str, str]]":
    """
    machine id = 0
    ag.c3 = 0.000000
    8:Width2
    """
    final: List[Dict[str, str]] = []
    kvp: Dict[str, str]
    for i in section_list:
        kvp = _parse_kvp(i, key_split)
        final.append(kvp)

    return final


def _parse_sample_header(section_list: "List[str]", key_split: str = ":") -> "List[Dict[str, str]]":
    """
    '0:ETG0187_0001_1  T=0001 L=1 P=1 D=1.000005 X=4.000000 H=ETG0187'
    '1:ETG0187_0001_2  T=0001 L=1 P=2 D=1.000006 X=12.000000 H=ETG0187'
    '2:ETG0187_0001_3  T=0001 L=1 P=3 D=1.000006 X=20.000000 H=ETG0187'
    """
    final: List[Dict[str, str]] = []
    key_0: str
    tmp_sample: Dict[str, str]

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


def _parse_class_section(section_list: "List[str]", classnumber: int) -> ClassHeaders:
    """
    name = S_jCLST_707 Groups
    max = 2
    colours = 15126526 30960
    0:SILICA
    1:K-FELDSPAR
    """
    class_names: Dict[str, str] = {}
    class_info: Dict[int, str] = {}
    c_name: str
    c_value: str
    for i in section_list:
        # lines of the section containing = form the class name
        if i.find("=") >= 0:
            split_i = i.split("=")
            c_name = split_i[0].strip()
            c_value = split_i[1].strip()
            class_names.update({c_name: c_value})
        elif i.find(":") >= 0:
            split_i = i.split(":")
            class_info.update({int(split_i[0]): split_i[1]})
    max_class: int = int(class_names["max"])
    class_header = ClassHeaders(classnumber, class_names["name"], max_class, class_info)
    return class_header


def _parse_wavelength_specs(line: str) -> "Dict[str, Union[float,str]]":
    split_wavelength: List[str] = line.split()
    wavelength = {
        "start": float(split_wavelength[0]),
        "end": float(split_wavelength[1]),
        "unit": split_wavelength[-1],
    }
    return wavelength


def _parse_kvp(line: str, split: str = "=") -> "Dict[str, str]":
    """Parses strings into Key value pairs
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


def _read_bip(filename: Union[str, Path], coordinates: "Dict[str, str]") -> NDArray[np.float32]:
    """Reads the .bip file as a 1d array then reshapes it according to the dimensions
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


def _calculate_wavelengths(wavelength_specs: "Dict[str,float]", coordinates: "Dict[str, str]") -> NDArray:
    wavelength_range: float = wavelength_specs["end"] - wavelength_specs["start"]
    resolution: float = wavelength_range / (int(coordinates["lastband"]) - 1)

    return np.arange(wavelength_specs["start"], wavelength_specs["end"] + resolution, resolution)


def read_hires_dat(filename: Union[str, Path]) -> NDArray:
    """Read the *hires.dat* file which contains the lidar scan
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


def _parse_bandheaders(bandheaders: "List[str]") -> "List[BandHeaders]":
    split_header = "List[str]"
    band: int
    info: str
    split_info: "List[str]"
    name: str
    class_name: int
    flag: int
    out: List[BandHeaders] = []
    for bh in bandheaders:
        split_header = bh.split(":")
        band = int(split_header[0])  # first item is the band number
        info = split_header[1]  # the second item is all the information
        split_info = info.split(";")
        name = split_info[0]
        if len(split_info) > 1:
            flag = int(split_info[3])
            if flag <= 2:
                class_name = int(split_info[4])
            elif flag == 13:
                # flag 13 is when PLS scalars are used
                class_name = split_info[4]
            else:
                class_name = float(split_info[4])

        else:
            class_name = -1
            flag = -1
        out.append(BandHeaders(band, name, class_name, flag))
    return out


def _parse_tsg(fstr: "List[str]", headers: "Dict[str, Tuple[int,int]]") -> "Dict[str, Any]":
    d_info: Dict[str, Any] = {}
    tmp_header: "List[Dict[str, str]]" = []
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
            header = _parse_bandheaders(fstr[start:end])
            d_info.update({k: header})
        elif k.find("class") == 0:
            class_number: int = int(k.split(" ")[1])  # this will be an int
            class_info = _parse_class_section(fstr[start:end], class_number)
            tmp_class = {class_number: class_info}

            if "class" in d_info.keys():
                d_info["class"].update(tmp_class)
            else:
                d_info.update({"class": tmp_class})

        else:
            tmp_out: Dict[str, str] = {}
            for i in fstr[start:end]:
                tmp = _parse_kvp(i)
                if not tmp is None:
                    tmp_out.update(tmp)
            d_info.update({k: tmp_out})

    return d_info


def _parse_scalars(
    scalars: NDArray, classes: "List[ClassHeaders]", bandheaders: "List[BandHeaders]"
) -> pd.DataFrame:
    """
    function to map the scalars to a pandas data frame with names and
    mapped values
    """
    tmp_series: List[pd.DataFrame] = []
    for i in bandheaders:
        band_value = scalars[:, i.band]
        # handle flag 13 that has a path to plsscalars
        if i.flag == 2:
            if i.class_number > 0:
                bv = band_value.astype(int)
                tn = classes[i.class_number].map_ints(bv)
                tmp_series.append(pd.DataFrame(tn, columns=[i.name]))
            else:
                tmp_series.append(pd.DataFrame(band_value, columns=[i.name]))
        else:
            tmp_series.append(pd.DataFrame(band_value, columns=[i.name]))
    output: pd.DataFrame = pd.concat(tmp_series, axis=1)

    return output


def read_tsg_bip_pair(tsg_file: Union[Path, str], bip_file: Union[Path, str], spectrum: str) -> Spectra:
    fstr = _read_tsg_file(tsg_file)
    headers = _find_header_sections(fstr)
    info = _parse_tsg(fstr, headers)
    spectra = _read_bip(bip_file, info["coordinates"])
    wavelength = _calculate_wavelengths(info["wavelength specs"], info["coordinates"])
    scalars = _parse_scalars(spectra[1, :, :], info["class"], info["band headers"])

    package = Spectra(
        spectrum,
        spectra[0, :, :],
        wavelength,
        info["sample headers"],
        info["class"],
        info["band headers"],
        scalars,
    )

    return package


def read_package(
    foldername: Union[str, Path],
    read_cras_file: bool = False,
    extract_cras: bool = False,
    imageoutput: Union[str, None] = None,
) -> TSG:
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

    file_pairs = FilePairs()
    files = foldername.glob("*.*")
    f: Path
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
        if extract_cras:
            if imageoutput is None:
                imageoutput = foldername.joinpath("IMG")

            if isinstance(imageoutput, str):
                imageoutput = Path(imageoutput)
            # check if the file exists
            if not imageoutput.exists():
                imageoutput.mkdir()

            extract_chips(file_pairs.cras, imageoutput, nir)
            cras = Cras
        else:
            cras = read_cras(file_pairs.cras)

    else:
        cras = Cras

    return TSG(nir, tir, cras, lidar)


if __name__ == "main":
    foldername = "data/RC_hyperspectral_geochem"
    results = read_package(foldername)
    pd.DataFrame(results.cras.section)
