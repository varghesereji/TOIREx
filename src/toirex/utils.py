#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import subprocess
import requests
import zipfile
import io
from astropy.io import fits
from scipy import signal
from scipy import ndimage
from astropy.io import ascii
from astropy.modeling import models, fitting

from WavelengthCalibrationTool import recalibrate

try:
    import importlib.resources as resources
except ImportError:
    import importlib_resources as resources

from ariastrotools import combine_process


def get_pkgpath():
    """
    Function to get the path
    to the package.
    """
    pkgpath = resources.files("toirex").joinpath(".")
    return pkgpath


def get_instrument_dir(instrument_name: str):
    """
    Ensures the instrument directory exists inside package data.

    If the directory does not exist, downloads it from the remote repository.

    Parameters:
    -----------
    instrument : str
        Name of the instrument directory (e.g., "TANSPEC", "TIRSPEC").
    """
    data_for = ['TANSPEC', 'TIRSPEC']
    if instrument_name not in data_for:
        print("No data for {}".format(instrument_name))
        return
    try:
        with resources.path("toirex.data", instrument_name) as p:
            if p.exists():
                return
    except FileNotFoundError:
        pass
    path = resources.files("toirex.data")
    try:
        print(f"[INFO] Downloading {instrument_name} data ....")
        download_instrument(instrument_name, path)
    except FileNotFoundError:
        print(f"Data for {instrument_name} is not added to the reposetory")


def download_instrument(instrument: str, outdir: Path = Path("data")) -> Path:
    """
    Download a whole instrument directory (e.g., TANSPEC) from GitLab repo.
    """
    url = "https://gitlab.com/varghesereji/toirex-data/-/archive/\
    main/toirex-data-main.zip"
    print("Downloading data for {}".format(instrument))
    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        members = [m for m in z.namelist() if f"/{instrument}/" in m]
        for member in members:
            z.extract(member, outdir)

    path = outdir / f"toirex-data-main/{instrument}"
    dest = Path(outdir) / path.name
    path.rename(dest)
    Path(path.parent).rmdir()


# Filename functions
def extract_number_from_fname(fname):
    numbers = re.findall(r"\d+", fname)
    return numbers


def extract_fname_prefix(fname, regexp=r"^(.*?)-\d{5}\.Z\.fits$"):
    match = re.match(regexp, fname)
    return match.group(1) if match else "AAA"


def text_to_dict(opdir='.',
                 txtfname="Filename_suggestions.txt"):
    if isinstance(opdir, str):
        opdir = Path(opdir)
    txtfile = opdir / txtfname
    text = txtfile.read_text().splitlines()
    mapping = {}
    for line in text:
        key, value = line.split(":", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def get_filename(groups, opdir):
    fname_suggestion = text_to_dict(
        txtfname=opdir / "Filename_suggestions.txt")[groups]
    outfileprefix = input(
        "Enter output filename prefix (default: {}) :".format(
            fname_suggestion)
    ) or fname_suggestion
    return outfileprefix


# Header functions
def read_fits_header(filename, ext=0):
    '''
    Function to read the header of the fits file.
    Input
    -----
    filename: Name of the fits file to read.
    ext: Extension of the fits file. Default 0
    '''
    header = fits.getheader(filename, ext=ext)
    return header


def read_fits_data(filename, ext=0):
    data = fits.getdata(filename, ext=ext)
    return data


def open_in_editor(path, config):
    """
    Open a text file in the desired text editor.
    """
    editor = config['inits']['EDITOR']
    subprocess.run([editor, str(path)])


def read_txt_file(filename):
    """
    Reads a text file and returns its contents as a list of lists of strings.

    Each line in the file is stripped of leading/trailing whitespace and split
    into components using spaces as delimiters.

    Parameters
    ----------
    filename : str
        Path to the text file to read.

    Returns
    -------
    list of list of str
        A list where each element corresponds to a line in the file, and
        each line is represented as a list of strings obtained by splitting
        on spaces.

    Example
    -------
    >>> read_txt_file("data.txt")
    [['123', 'abc'], ['456', 'def']]
    """
    txtline = []
    with open(filename, 'r') as txtfile:
        for line in txtfile:
            txtline.append(line.strip().split(" "))
    return txtline


def write_asciitable(content,
                     fname,
                     headers,
                     format='basic',
                     delimiter="|"):
    ascii.write(list(zip(*content)),
                fname,
                names=headers,
                format=format,
                delimiter=delimiter,
                overwrite=True)


def combine_frames(files_list, op_dirname, sorting_function,
                   method='median',
                   op_prefix="Comb_",
                   fluxext=0,
                   varext=1,
                   mask=None):
    fnums = []
    if "".join(varext) == 'None':
        varext = None
    for fname in files_list:
        fnum = sorting_function(fname)
        fnums.append(fnum)
    comb_fnums = "_".join([str(n) for n in fnums])
    comb_filename = op_prefix + "{}.fits".format(comb_fnums)
    data_dirname = Path(op_dirname.name)
    targets_path = [data_dirname / frame for frame in files_list]
    comb_fname = op_dirname / comb_filename
    if comb_fname.exists():
        return comb_fname.name
    combine_process(targets_path,
                    comb_fname,
                    method='biweight',
                    fluxext=fluxext,
                    varext=varext,
                    mask=mask)
    return comb_fname.name


def get_pixel_shift(spectra, template, medfilt=3, sigma=10, radius=20):
    """
    Function to find pixel offset between two spectra.
    """
    arc_filtered = ndimage.gaussian_filter(signal.medfilt(spectra, medfilt),
                                           sigma=10, radius=20)
    pixelshift = recalibrate.calculate_pixshift_with_phase_cross_correlation(
        template, arc_filtered)
    return pixelshift


def fit_gaussian_profile(counts):
    x = np.arange(len(counts))

    # Initial guess: amplitude, mean, stddev
    amplitude_guess = np.max(counts) - np.min(counts)
    mean_guess = np.argmax(counts)
    stddev_guess = len(counts) / 4

    g_init = models.Gaussian1D(amplitude=amplitude_guess,
                               mean=mean_guess,
                               stddev=stddev_guess)
    fit_g = fitting.LevMarLSQFitter()

    g_fit = fit_g(g_init, x, counts)

    return x, counts, g_fit(x), g_fit
# End
