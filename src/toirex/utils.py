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
    """
    Extract the prefix from a FITS filename using a regular expression.

    By default, the function extracts the portion of the filename
    preceding a five-digit sequence and the ``.Z.fits`` suffix. If the
    filename does not match the supplied pattern, ``"AAA"`` is returned.

    Parameters
    ----------
    fname : str
        Filename from which to extract the prefix.
    regexp : str, optional
        Regular expression used to match the filename. The first capture
        group is returned as the prefix. Default is
        ``r"^(.*?)-\\d{5}\\.Z\\.fits$"``.

    Returns
    -------
    str
        Extracted filename prefix if the pattern matches; otherwise
        ``"AAA"``.

    Examples
    --------
    >>> extract_fname_prefix("NGC7027-00001.Z.fits")
    'NGC7027'

    >>> extract_fname_prefix("invalid_filename.fits")
    'AAA'
    """
    match = re.match(regexp, fname)
    return match.group(1) if match else "AAA"


def text_to_dict(opdir='.',
                 txtfname="Filename_suggestions.txt"):
    """
    Read a key-value mapping from a text file.

    Each non-empty line in the input file must contain a key and value
    separated by the first colon (``:``). Leading and trailing
    whitespace is removed from both the key and value before storing
    them in the returned dictionary.

    Parameters
    ----------
    opdir : str or pathlib.Path, optional
        Directory containing the text file. Default is the current
        directory (``"."``).
    txtfname : str or pathlib.Path, optional
        Name of the text file to read. Default is
        ``"Filename_suggestions.txt"``.

    Returns
    -------
    mapping : dict
        Dictionary containing the parsed key-value pairs.

    Notes
    -----
    Only the first colon in each line is treated as the separator,
    allowing values to contain additional colons.
    """
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
    """
    Prompt the user for an output filename prefix.

    A default filename prefix is obtained from the
    ``Filename_suggestions.txt`` file located in the output directory.
    The user may either accept the suggested name by pressing Enter or
    provide a custom prefix.

    Parameters
    ----------
    groups : str
        Key identifying the group for which a filename suggestion should
        be retrieved.
    opdir : str or pathlib.Path
        Output directory containing the ``Filename_suggestions.txt``
        file.

    Returns
    -------
    outfileprefix : str
        User-selected output filename prefix. If no input is provided,
        the suggested prefix is returned.

    Notes
    -----
    Filename suggestions are read using :func:`text_to_dict`.
    """
    fname_suggestion = text_to_dict(
        txtfname=opdir / "Filename_suggestions.txt")[groups]
    outfileprefix = input(
        "Enter output filename prefix (default: {}) :".format(
            fname_suggestion)
    ) or fname_suggestion
    return outfileprefix


# Header functions
def read_fits_header(filename, ext=0):
    """
    Read the header from a FITS file extension.

    This is a convenience wrapper around
    :func:`astropy.io.fits.getheader` for reading the header of a
    specified FITS extension.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS file.
    ext : int or str, optional
        FITS extension from which to read the header. This may be either
        the extension number or the extension name. Default is 0.

    Returns
    -------
    header : `astropy.io.fits.Header`
        Header associated with the specified FITS extension.

    See Also
    --------
    astropy.io.fits.getheader
        Read the header from a FITS file.
    """
    header = fits.getheader(filename, ext=ext)
    return header


def read_fits_data(filename, ext=0):
    """
    Read data from a FITS file extension.

    This is a convenience wrapper around :func:`astropy.io.fits.getdata`
    for reading the data array from a specified FITS extension.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the FITS file.
    ext : int or str, optional
        FITS extension from which to read the data. This may be either
        the extension number or the extension name. Default is 0.

    Returns
    -------
    data : `numpy.ndarray`
        Data array stored in the specified FITS extension.

    See Also
    --------
    astropy.io.fits.getdata
        Read data from a FITS file.
    """
    data = fits.getdata(filename, ext=ext)
    return data


def open_in_editor(path, config):
    """
    Open a file in the user-configured text editor.

    The editor executable is read from the ``EDITOR`` entry in the
    ``inits`` section of the configuration dictionary and invoked using
    :func:`subprocess.run`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file to be opened.
    config : dict
        Configuration dictionary containing the editor setting. The
        editor executable is expected at::

            config['inits']['EDITOR']

    Returns
    -------
    None

    Notes
    -----
    This function launches the configured editor as a subprocess and
    waits for it to exit before returning.
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
    """
    Write tabular data to an ASCII file.

    This is a convenience wrapper around :func:`astropy.io.ascii.write`.
    The input columns are transposed into row-oriented data before being
    written to disk.

    Parameters
    ----------
    content : sequence of array-like
        Column-wise data to be written. Each element represents a column
        in the output table and must have the same length.
    fname : str or pathlib.Path
        Name or path of the output ASCII table.
    headers : sequence of str
        Column names for the output table.
    format : str, optional
        Output table format understood by
        :func:`astropy.io.ascii.write`. Default is ``'basic'``.
    delimiter : str, optional
        Column delimiter used in the output file. Default is ``"|"``.

    Returns
    -------
    None

    Notes
    -----
    Existing files are overwritten by default by passing
    ``overwrite=True`` to :func:`astropy.io.ascii.write`.
    """
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
    """
    Combine multiple FITS frames into a single output frame.

    The input filenames are first sorted into a descriptive output name
    using the provided ``sorting_function``. If the combined frame already
    exists, its filename is returned without recomputing the combination.
    Otherwise, the input frames are combined using
    :func:`combine_process`.

    Parameters
    ----------
    files_list : list of str
        List of FITS filenames to be combined.
    op_dirname : pathlib.Path
        Directory containing the input frames and where the combined
        output frame will be written.
    sorting_function : callable
        Function that accepts a filename and returns a value (typically a
        frame number) used to construct the output filename.
    method : {'median', 'mean', 'biweight'}, optional
        Combination method to use. Default is ``'median'``.

        .. note::
           The current implementation always calls
           :func:`combine_process` with ``method='biweight'``,
           regardless of the value supplied to this parameter.
    op_prefix : str, optional
        Prefix for the output filename. Default is ``"Comb_"``.
    fluxext : int, optional
        FITS extension containing the science (flux) data. Default is 0.
    varext : int or None, optional
        FITS extension containing the variance data. If ``None``, no
        variance extension is used. Default is 1.
    mask : array-like or None, optional
        Boolean mask specifying pixels to ignore during the combination.
        Default is ``None``.

    Returns
    -------
    str
        Filename of the combined FITS frame. If the combined file already
        exists, the existing filename is returned.

    Notes
    -----
    The output filename is constructed by concatenating the values
    returned by ``sorting_function`` for each input file, separated by
    underscores. For example::

        Comb_001_002_003.fits

    If the output file already exists, no combination is performed.
    """
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
    Determine the pixel shift between an observed spectrum and a template.

    The input spectrum is first median filtered to suppress impulsive
    noise and then smoothed with a Gaussian filter. The pixel offset
    between the processed spectrum and the template is computed using
    phase cross-correlation.

    Parameters
    ----------
    spectra : array-like
        One-dimensional observed spectrum.
    template : array-like
        One-dimensional reference spectrum used as the template.
    medfilt : int, optional
        Kernel size for the median filter. Default is 3.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel. Default is 10.
    radius : int, optional
        Radius of the Gaussian kernel. Default is 20.

    Returns
    -------
    pixelshift : float
        Estimated pixel shift between the input spectrum and the template.
        A positive value indicates that the observed spectrum is shifted
        towards higher pixel indices relative to the template.

    Notes
    -----
    The spectrum is preprocessed as follows before computing the shift:

    1. Median filtering using :func:`scipy.signal.medfilt`.
    2. Gaussian smoothing using :func:`scipy.ndimage.gaussian_filter`.
    3. Phase cross-correlation using
       :func:`recalibrate.calculate_pixshift_with_phase_cross_correlation`.
    """
    arc_filtered = ndimage.gaussian_filter(signal.medfilt(spectra, medfilt),
                                           sigma=10, radius=20)
    pixelshift = recalibrate.calculate_pixshift_with_phase_cross_correlation(
        template, arc_filtered)
    return pixelshift


def fit_gaussian_profile(counts):
    """
    Fit a one-dimensional Gaussian profile to a sequence of counts.

    This function fits an `astropy.modeling.models.Gaussian1D` model to
    the input data using a Levenberg-Marquardt least-squares optimizer.
    Initial parameter estimates are derived from the input profile:

    - Amplitude: ``max(counts) - min(counts)``
    - Mean: Index of the maximum value
    - Standard deviation: One-quarter of the profile length

    Parameters
    ----------
    counts : array-like
        One-dimensional array containing the profile values to be fitted.

    Returns
    -------
    x : `numpy.ndarray`
        Pixel indices corresponding to the input profile.
    counts : array-like
        The original input profile.
    fitted_profile : `numpy.ndarray`
        Gaussian model evaluated at ``x`` using the best-fit parameters.
    g_fit : `astropy.modeling.functional_models.Gaussian1D`
        Best-fit Gaussian model. The fitted parameters can be accessed
        through attributes such as ``g_fit.amplitude``,
        ``g_fit.mean``, and ``g_fit.stddev``.

    Notes
    -----
    The fitting is performed using
    `astropy.modeling.fitting.LevMarLSQFitter`, which minimizes the
    least-squares residuals between the input profile and the Gaussian
    model.
    """
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
