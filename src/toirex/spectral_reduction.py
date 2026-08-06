#!/usr/bin/env python3
"""
Spectral extraction and calibration routines.

This module provides the high-level spectral reduction workflow for TOIREx.
It interfaces with SpectrumExtractor to extract one-dimensional spectra,
performs wavelength calibration using arc-lamp spectra, optionally subtracts
the sky background, applies instrument response correction, and combines
multiple reduced spectra when requested.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections.abc import Callable
from typing import Any

from ariastrotools import combine_process
from ariastrotools import combine_spectra
from ariastrotools import operate_process
import SpectrumExtractor.spectrum_extractor as specextractor
from WavelengthCalibrationTool import recalibrate

from astropy.io import fits

from .setups import get_logger

from .instrument import instruments
from .utils import get_pkgpath
from .utils import read_txt_file
from .setups import read_config
from .setups import create_config
from .plottings import plot_arrays


def config_for_extraction(
        data_fname: str | Path,
        config: dict[str, Any],
        trace_selection: Callable[[Path], tuple[Path, Path, Path]],
        for_lamp: dict[str, Any] | None = None):
    """
    Create a SpectrumExtractor configuration file for a science or lamp frame.

    This function prepares a temporary extraction configuration by combining
    the pipeline configuration with a SpectrumExtractor configuration template.
    If no extractor configuration is specified, the default configuration
    distributed with the package is used. The function also updates tracing and
    extraction parameters, selects the appropriate aperture trace, and writes
    the resulting configuration to disk.

    Parameters
    ----------
    data_fname : str or pathlib.Path
        Path to the input FITS file for which the extraction configuration
        should be generated.
    config : dict
        Pipeline configuration dictionary containing the spectral extraction
        parameters and the location of the SpectrumExtractor configuration
        template.
    trace_selection : callable
        Function that selects the appropriate aperture trace for the input
        frame. It must accept ``data_fname`` as input and return a tuple
        ``(continuum_file, aperture_label, aperture_trace)``.
    for_lamp : dict, optional
        Dictionary containing tracing parameters for lamp extraction. If
        provided, a lamp-specific configuration is generated instead of a
        science-frame configuration. The dictionary is expected to contain
        the key ``"ReFitApertureInXD"``.

    Returns
    -------
    pathlib.Path
        Path to the generated SpectrumExtractor configuration file.

    Notes
    -----
    - If ``config['spectral_extraction']['EXTRACTORCONFIG']`` is empty or
      set to ``"default"``, the default configuration bundled with the
      package is used.
    - For manual aperture selection
      (``SELECT_APERTURE == 'MANUAL'``), the aperture label and trace files
      are assumed to have the same basename as the input FITS file with
      ``.npy`` and ``.pkl`` extensions, respectively.
    - The generated configuration file is written in the same directory as
      the input FITS file. Science-frame configurations are named
      ``<filename>.config``, while lamp configurations are named
      ``<filename>.Lamp.config``.

    See Also
    --------
    read_config : Read a SpectrumExtractor configuration file.
    create_config : Write a SpectrumExtractor configuration file.
    get_pkgpath : Return the installation path of the package.
    """
    data_fname = Path(data_fname)
    dirname = Path(data_fname.parent)
    # opdir = Path(config['outputs']['OP_DIR']) / dirname
    extractorconfig_fname = config['spectral_extraction']['EXTRACTORCONFIG']
    defaultconfig = False
    # If user does not specify any config file for spectral extraction,
    # uses default config and traces. The traces will be specific for
    # each instrument.
    if (len(extractorconfig_fname) == 0) or (
            extractorconfig_fname.lower() == 'default'):
        defaultconfig = True
        print('\n \033[1;32m Uses default config file\033[0m' +
              '\033[1;32m (https://github.com/varghesereji/config/' +
              'spectrum_extractor.config)'
              + ' for spectrum extraction' + '\033[0m')
        extractorconfig_fname = get_pkgpath() / \
            'config/spectrum_extractor.config'

    extraction_config = read_config(extractorconfig_fname)
    tracing_settings = extraction_config['tracing_settings']
    extraction_mode = config['spectral_extraction']['SELECT_APERTURE']
    if defaultconfig:
        # Taking trace saved with pipeline
        if extraction_mode == 'MANUAL':
            print("Manual selection of aperture trace")
            star_trace = data_fname
            aperture_label = str(data_fname)[:-5] + ".npy"
            aperturetrace = str(data_fname)[:-5] + ".pkl"
            tracing_settings['Mode'] = 'MANUAL'
        else:
            star_trace, aperture_label, aperturetrace = trace_selection(
                data_fname
            )
        tracing_settings['ContinuumFile'] = str(star_trace)
        tracing_settings['ApertureLabel'] = str(aperture_label)
        tracing_settings['ApertureTraceFilename'] = str(aperturetrace)

        # Tracing settings from config file
        tracing_settings['ReFitApertureInXD'] = config[
            'spectral_extraction'
        ][
            'REFITAPERTUREINXD'
        ]
        tracing_settings['ReFitApertureInXD_DWindow'] = config[
            'spectral_extraction'
        ][
            'REFITAPERTUREINXD_DWINDOW'
        ]
        tracing_settings['ReFitApertureInXD_BkgMedianFilt'] = config[
            'spectral_extraction'
        ][
            'REFITAPERTUREINXD_BKGMEDIANFILT'
        ]
        tracing_settings['DCoeffModelForAperReFit'] = config[
            'spectral_extraction'
        ][
            'DCOEFFMODELFORAPERREFIT'
        ]

    # Setting up aperture windows
    extraction_settings = extraction_config['extraction_settings']
    aperturewindow = config['spectral_extraction']['APERTUREWINDOW']
    bkgwindow = config['spectral_extraction']['BKGWINDOWS']
    extraction_settings['ApertureWindow'] = aperturewindow
    extraction_settings['BkgWindows'] = bkgwindow

    # Creating new configfile
    if for_lamp is None:
        # Creating the config file for extraction from science frame
        new_configfname = Path(data_fname).stem + ".config"
    else:
        # Creating the config file for lamps
        new_configfname = Path(data_fname).stem + ".Lamp.config"
        tracing_settings["ReFitApertureInXD"] = str(
            for_lamp["ReFitApertureInXD"]
        )
        tracing_settings["ShowPlot_Trace"] = str(False)

    new_configfname = dirname / new_configfname

    create_config(new_configfname, extraction_config)
    return new_configfname


# ---------------------------
# Background subtraction
# ---------------------------

def subtract_background(
        fname: str | Path,
        config: dict[str, Any]):
    """
    Subtract the estimated background from an extracted spectrum.

    This function computes the average background per pixel using the two
    background regions defined in the pipeline configuration and subtracts
    the corresponding background contribution from the extracted flux. If a
    variance extension is present, the variance is propagated assuming the
    background estimates are independent.

    The input FITS file is modified in place.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the extracted FITS file. The primary HDU must contain the
        extracted flux, while extensions 1 and 2 must contain the summed
        background values from the two background windows.
    config : dict
        Pipeline configuration dictionary containing the spectral extraction
        parameters, including ``APERTUREWINDOW`` and ``BKGWINDOWS``.

    Returns
    -------
    None

    Notes
    -----
    - The average background per pixel is computed independently for the two
      background windows and then averaged.
    - The total background within the extraction aperture is obtained by
      scaling the average background by the aperture width before subtraction.
    - If the FITS file contains a ``VARIANCE`` extension, the corresponding
      background variance extensions (``BKG VARIANCE 0`` and
      ``BKG VARIANCE 1``) are used to propagate the uncertainties.
    - A HISTORY entry describing the aperture and background windows used for
      the subtraction is added to the primary FITS header.
    - The FITS file is updated in place.

    See Also
    --------
    astropy.io.fits.open : Open a FITS file for reading or updating.
    """
    logger = get_logger("spectral")
    logger.info(f"Background subtraction on: {str(fname)}")
    hdul = fits.open(fname, mode='update')
    flux = hdul[0].data
    bkg1 = hdul[1].data
    bkg2 = hdul[2].data

    aperture_window = ast.literal_eval(
        config['spectral_extraction']['APERTUREWINDOW']
        )
    bkg_window = ast.literal_eval(
        config['spectral_extraction']['BKGWINDOWS']
        )
    logger.info(f"Aperture size: {aperture_window}")
    logger.info(f"Background size: {bkg_window}")
    aperture_width = float(aperture_window[1]) - float(aperture_window[0])
    bkg_width1 = float(bkg_window[0][1]) - float(bkg_window[0][0])
    bkg_width2 = float(bkg_window[1][1]) - float(bkg_window[1][0])

    # Calculating average bkg per pixel
    avgbkg = ((bkg1 / bkg_width1) + (bkg2 / bkg_width2)) / 2
    aperture_bkg = avgbkg * aperture_width

    # print(aperture_width,)
    clear_flux = flux - aperture_bkg
    hdul[0].data = clear_flux

    # Error propagation, if variance exists
    ext_name = "VARIANCE"
    exists = any(hdu.name == ext_name for hdu in hdul)
    if exists:
        fluxvar = hdul['VARIANCE'].data
        bkgvar_0 = hdul['BKG VARIANCE 0'].data
        bkgvar_1 = hdul['BKG VARIANCE 1'].data

        scale_bkgvar_1 = bkgvar_0 / bkg_width1 ** 2
        scale_bkgvar_2 = bkgvar_1 / bkg_width2 ** 2

        comb_bkgvar = (scale_bkgvar_1 + scale_bkgvar_2) / 4
        clear_fluxvar = fluxvar + comb_bkgvar * aperture_width**2
        hdul['VARIANCE'].data = clear_fluxvar

    hdr = hdul[0].header
    history = "Subtracted background using aperture window {}\
    and bkg window {}."
    hdr.add_history(
        history.format(
            aperture_window, bkg_window))

    hdul.flush()
    hdul.close()

# ---------------------------
# Wavelength calibration
# ---------------------------


def wavelength_calibration(
        txtline: list[str],
        config: dict[str, Any],
        opdir: Path,
        instrument: dict[str, Callable[..., Any] | None],
) -> str:
    """
    Derive and save the wavelength solution for an extracted spectrum.

    This function calibrates the wavelength solution for each extracted
    aperture by matching the observed arc-lamp spectrum to an instrument-
    specific template. If multiple lamp exposures are provided, they are
    first combined into a single reference lamp frame. The resulting
    wavelength solution is appended as a new FITS extension and written to
    a new output file.

    Diagnostic plots showing the template match for each aperture are also
    generated.

    Parameters
    ----------
    txtline : list of str
        List containing the science spectrum filename followed by one or more
        arc-lamp filenames. The first element is the extracted science FITS
        file, while the remaining elements are the associated calibration
        lamp frames.
    config : dict
        Pipeline configuration dictionary. Currently included for interface
        consistency and reserved for future use.
    opdir : pathlib.Path
        Directory containing the extracted science and lamp files. Output
        files are also written to this directory.
    instrument : dict
        Instrument-specific configuration dictionary. It must contain:

        - ``'pixel_offset'`` : callable or ``None``
          Function that estimates the detector pixel offset from a lamp frame.
        - ``'get_template'`` : callable
          Function returning the reference wavelength template for a given
          aperture.

    Returns
    -------
    str
        Filename of the generated wavelength-calibrated FITS file.

    Notes
    -----
    - When multiple lamp frames are supplied, they are combined using a mean
      before wavelength calibration.
    - The wavelength solution for each aperture is determined using
      ``recalibrate.ReCalibrateDispersionSolution`` with a third-order
      polynomial model (``method='p3'``).
    - One diagnostic PDF,
      ``template_match_aperture<N>.pdf``, is generated for each aperture,
      showing the observed lamp spectrum and the matched template.
    - The computed wavelength solution is stored as a new FITS extension
      named ``"Wavelength"``.
    - The output file is written with the suffix ``.wlc.fits``.

    See Also
    --------
    combine_process : Combine multiple lamp exposures.
    recalibrate.ReCalibrateDispersionSolution
        Fit the wavelength solution using a reference template.
    instrument['get_template']
        Return the reference wavelength template for an aperture.
    """
    logger = get_logger("spectral")
    op_fname = opdir / txtline[0]
    arclamp1 = opdir / txtline[1]
    logger.info(f"Wavelength calibration on {op_fname}")
    calculate_pixel_offset = instrument['pixel_offset']
    if calculate_pixel_offset is None:
        offset = 0
    else:
        offset = calculate_pixel_offset(arclamp1)
    logger.info(f"Pixel offset calculated: {offset}")
    if len(txtline) > 1:
        comb_lampname = Path(op_fname).stem + "_combarc.fits"
        comb_lampname = opdir / comb_lampname
        lamps_list = [opdir / i for i in txtline[1:]]
        logger.info(f"Combining lamps {lamps_list}")
        logger.info(f"Saved as {comb_lampname}")
        combine_process(lamps_list,
                        comb_lampname,
                        method='mean',
                        fluxext=[0],
                        varext=[1])
    else:
        comb_lampname = arclamp1
    hdu_arcdata = fits.getdata(comb_lampname, ext=0)
    wlsoln = None
    for index, lampflux in enumerate(hdu_arcdata):
        logger.info(f"Wavelength calibration on Aperture {index}")
        lamp = hdu_arcdata[index]
        template = instrument['get_template'](comb_lampname, index)
        soln, shift = recalibrate.ReCalibrateDispersionSolution(
            lamp,
            template.T,
            method='p3',
            initial_guess=[1,
                           -offset * np.median(
                               np.gradient(
                                   template.T[:, 0])
                           )*2/(max(template.T[:, 0])-min(template.T[:, 0])),
                           1, 0, 0]
            )
        plt.figure()
        plt.plot(soln, lamp/np.max(lamp), label='Observed lamp')
        plt.plot(template[0], template[1] / np.max(template[1]),
                 label='Template')
        plt.legend()
        template_match_filename = opdir / \
            'template_match_aperture{}.pdf'.format(index)
        plt.title('Aperture {}'.format(index))
        plt.savefig(template_match_filename)
        plt.close()
        if wlsoln is None:
            wlsoln = soln
        else:
            wlsoln = np.vstack((wlsoln, soln))
    # Make sure the wavelength solution is always 2D array
    if wlsoln.ndim == 1:
        wlsoln = wlsoln.reshape(1, -1)
    # Saving wavelength solution with result

    op_hdul = fits.open(op_fname)
    wlsoln_hdu = fits.ImageHDU(wlsoln, name="Wavelength")
    op_hdul.append(wlsoln_hdu)
    soln_fname = op_fname.stem + ".wlc.fits"
    soln_fname = opdir / soln_fname
    op_hdul.writeto(soln_fname)
    logger.info(f"WL calibrated file Saved as {str(soln_fname)}")
    return soln_fname.name

# ---------------------------
# Flux calibration
# ---------------------------


def flux_calibration(
        fname: str | Path,
        config: dict[str, Any],
        instrument: dict[str, Callable[..., Any] | None]
) -> Path:
    """
    Apply instrumental response correction to an extracted spectrum.

    This function performs flux calibration by dividing the extracted spectrum
    by the instrument response curve. The appropriate response file is obtained
    using the instrument-specific response function, and both the flux and
    variance extensions are calibrated accordingly.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the extracted FITS spectrum to be flux calibrated.
    config : dict
        Pipeline configuration dictionary. The ``inputs`` section must define
        the FITS extensions corresponding to the flux and variance data
        (``FLUXEXT`` and ``VAREXT``).
    instrument : dict
        Instrument-specific configuration dictionary. It must contain the key
        ``'inst_response'``, which is a callable that accepts ``fname`` and
        returns the path to the corresponding instrument response file.

    Returns
    -------
    pathlib.Path
        Path to the flux-calibrated FITS file.

    Notes
    -----
    - Flux calibration is performed by dividing the extracted spectrum by the
      instrument response curve.
    - The same operation is applied to the specified variance extension so
      that the propagated uncertainties remain consistent.
    - The output file is written to the same directory as the input spectrum
      with the suffix ``.flc.fits``.

    See Also
    --------
    operate_process
        Perform arithmetic operations between FITS files while propagating
        variances.
    """
    logger = get_logger("spectral")
    response_name = instrument['inst_response'](fname)
    print(
        "Doing flux calibration with response curves: {}".format(
            response_name)
    )
    logger.info("Flux calibration")
    opname = fname.stem + ".flc.fits"
    opname = Path(fname.parent) / opname
    logger.info(f"{str(fname)} / {response_name} = {str(opname)}")
    operate_process(str(fname), str(response_name),
                    opfilename=opname,
                    operation='/',
                    fluxext=[config['inputs']['FLUXEXT']],
                    varext=[config['inputs']['VAREXT']])
    print("Flux calibrated spectra: {}".format(opname))
    return opname


# Plot sky

def plot_sky(
        fname: str | Path,
        opdir: Path,
        getsky_fn: Callable[[Path], Path] | None,
) -> None:
    """
    Plot the extracted sky background together with a reference sky spectrum.

    This function plots the two extracted background spectra from an
    observation and, if available, overlays a standard sky spectrum for
    comparison. All spectra are normalized by their median values before
    plotting. The resulting figure is saved as a PDF in the output directory.

    Parameters
    ----------
    fname : str or pathlib.Path
        Filename of the extracted FITS spectrum.
    opdir : pathlib.Path
        Directory containing the extracted spectrum and where the output plot
        will be saved.
    getsky_fn : callable or None
        Function that accepts the extracted spectrum filename and returns the
        path to the corresponding standard sky spectrum. If ``None``, the
        function returns without producing any plot.

    Returns
    -------
    None

    Notes
    -----
    - The extracted background spectra are read from the ``"BKG FLUX 0"`` and
      ``"BKG FLUX 1"`` FITS extensions.
    - The wavelength scale is read from the ``"WAVELENGTH"`` extension.
    - All spectra are normalized by their median values to facilitate
      comparison of their spectral shapes.
    - If the standard sky spectrum cannot be found, only the extracted
      background spectra are plotted and no output file is written.
    - When available, the comparison plot is saved as
      ``<filename>_stdsky.pdf`` in the output directory.

    See Also
    --------
    plot_arrays
        Plot one or more spectra on a common set of axes.
    """
    if getsky_fn is None:
        return
    fname = opdir / fname
    hdul = fits.open(fname)
    bkg1 = hdul['BKG FLUX 0'].data
    bkg2 = hdul['BKG FLUX 1'].data
    bkg1 = bkg1 / np.nanmedian(bkg1)
    bkg2 = bkg2 / np.nanmedian(bkg2)
    wl = hdul['WAVELENGTH'].data
    hdul.close()
    fig, axs = plot_arrays(wl, bkg1, title="Background",
                           label="Bkg 1", show=False)
    plot_arrays(wl, bkg2, fig_axs=(fig, axs), label="Bkg 2", show=False)
    stdsky_fname = getsky_fn(fname)
    try:
        hdul_std = fits.open(stdsky_fname)
    except FileNotFoundError:
        print("Standard sky does not exist")
    else:
        sky_std = hdul_std[0].data
        sky_std = sky_std / np.nanmedian(sky_std)
        wl_std = hdul_std[1].data
        plot_arrays(wl_std, sky_std, fig_axs=(fig, axs), label="Std sky",
                    show=False)
        plot_name = fname.stem + "_stdsky.pdf"
        plot_name = opdir / plot_name
        fig.savefig(plot_name)
    plt.close()


# --------------------- #
#  Spectral Extraction  #
# --------------------- #


def extraction(
        fname: str | Path,
        extraction_config: str | Path,
        op_fname: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a spectrum from a reduced two-dimensional image.

    This function runs the SpectrumExtractor package using the supplied
    extraction configuration file to produce an extracted one-dimensional
    spectrum. If an output filename is not provided, a default filename is
    created in the same directory as the input image.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the reduced two-dimensional FITS image from which the spectrum
        is to be extracted.
    extraction_config : str or pathlib.Path
        Path to the SpectrumExtractor configuration file containing the
        tracing and extraction parameters.
    op_fname : str or pathlib.Path, optional
        Path to the output FITS file. If not provided, the extracted spectrum
        is written to ``<input_stem>.ms.fits`` in the same directory as the
        input file.

    Returns
    -------
    outputobjspec : numpy.ndarray
        Extracted one-dimensional spectrum returned by SpectrumExtractor.
    avg_xd_shift : numpy.ndarray
        Cross-dispersion shift(s) measured during extraction.
    pixdomain : numpy.ndarray
        Pixel coordinates corresponding to the extracted spectrum.

    Notes
    -----
    - The extraction is performed by calling ``specextractor.main``.
    - The extracted spectrum is written to the specified output FITS file.
    - If ``fname`` is supplied as a string, it is converted to a
      ``pathlib.Path`` object before processing.

    See Also
    --------
    specextractor.main
        Perform spectral extraction using the SpectrumExtractor package.
    """
    logger = get_logger("spectral")
    if isinstance(fname, str):
        fname = Path(fname)
    opdir = Path(fname.parent)
    if op_fname is None:
        op_fname = fname.stem + ".ms.fits"
        op_fname = opdir / op_fname
    print("Extracting spectrum from", fname)
    logger.info(f"Extracting spectra from {fname}")
    outputobjspec, avg_xd_shift, pixdomain = specextractor.main(
        [str(fname),
         str(extraction_config),
         str(op_fname)]
    )
    logger.info(f"output file name: {str(op_fname)}")
    logger.info(f"Config used: {str(extraction_config)}")
    return outputobjspec, avg_xd_shift, pixdomain


def extract_obj_lamp(
        txtline: list[str],
        config: dict,
        opdir: Path,
        trace_selection: Callable,
) -> list[str]:
    """
    Extract spectra from a science frame and its associated arc-lamp frames.

    This function extracts the spectrum from a science exposure and then
    extracts the spectra from one or more associated arc-lamp exposures using
    the same aperture traces. The measured cross-dispersion shifts from the
    science extraction are reused to refit the aperture positions for the lamp
    frames, ensuring that the wavelength calibration is performed using
    consistent extraction apertures.

    Parameters
    ----------
    txtline : list of str
        List containing the science filename followed by one or more
        associated arc-lamp filenames. The first element is treated as the
        science frame, while the remaining elements are extracted as lamp
        spectra.
    config : dict
        Pipeline configuration dictionary containing the spectral extraction
        parameters.
    opdir : pathlib.Path
        Directory containing the input files and where the extracted spectra
        will be written.
    trace_selection : callable
        Function that selects the appropriate aperture trace for the
        science frame.

    Returns
    -------
    list of str
        List of output filenames. The first element is the extracted science
        spectrum, followed by the extracted lamp spectra in the same order as
        the input lamp filenames.

    Notes
    -----
    - The science spectrum is extracted first using the configured tracing
      settings.
    - The average cross-dispersion shifts and pixel domains measured during
      the science extraction are reused when generating the extraction
      configuration for the lamp frames.
    - Each lamp spectrum is extracted using the same aperture geometry as the
      science spectrum.
    - By default, the science spectrum is written as
      ``<science>.ms.fits`` and the lamp spectra as
      ``<science>.ms_arc<N>.fits``, where ``<N>`` is the lamp index.

    See Also
    --------
    config_for_extraction
        Generate the SpectrumExtractor configuration file.
    extraction
        Extract a one-dimensional spectrum from a two-dimensional image.
    """
    logger = get_logger("spectral")
    data_fname = opdir / txtline[0]

    extraction_config = config_for_extraction(data_fname,
                                              config,
                                              trace_selection)
    op_fname = Path(data_fname).stem + ".ms.fits"
    op_fname = Path(opdir) / op_fname
    optxtfile_line = [op_fname.name]
    _, avg_xd_shift, pixdomain = extraction(data_fname,
                                            extraction_config)
    refit_aperture_in_xd = (
        tuple(
            x.item() if isinstance(x, np.generic) else x for x in avg_xd_shift
        ),
        tuple(
            x.item() if isinstance(x, np.generic) else x for x in pixdomain
        ),
    )
    logger.info(f"Refit aperture: {refit_aperture_in_xd}")
    # Extracting lamps
    # We want to extract the lamp spectra from the same place where
    # science spectra was extracted.
    lamp_entries = {"ReFitApertureInXD": refit_aperture_in_xd}
    config['spectral_extraction']['EXTRACTORCONFIG'] = str(extraction_config)
    lamp_config = config_for_extraction(data_fname,
                                        config,
                                        trace_selection,
                                        for_lamp=lamp_entries
                                        )
    logger.info(f"Config for lamp: {lamp_config}")
    lamp_fnames = txtline[1:]

    for n, lamps in enumerate(lamp_fnames):
        lampfile = opdir / lamps
        outlamp_fname = Path(op_fname).stem + "_arc{}.fits".format(n+1)
        optxtfile_line.append(outlamp_fname)
        outlamp_fname = opdir / outlamp_fname
        _, _, _ = extraction(lampfile,
                             lamp_config,
                             outlamp_fname)
    return optxtfile_line


# --------------------------------------- #
# Reduction with wavelength calibration   #
# --------------------------------------- #
def spectral_reduction(
        config: dict,
        dirname: str | Path
) -> None:
    """
    Perform spectral extraction and calibration for a group of observations.

    This function executes the complete spectral reduction workflow for all
    observation groups in a reduced data directory. For each science frame, it
    extracts the science and arc-lamp spectra, performs wavelength
    calibration, optionally subtracts the background and applies flux
    calibration, and generates diagnostic sky plots. If multiple spectra are
    available within a group, they can optionally be combined into a single
    averaged spectrum.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary containing the reduction,
        extraction, calibration, and output parameters.
    dirname : str or pathlib.Path
        Relative directory containing the reduced observations. The full
        output directory is constructed using
        ``config['outputs']['OP_DIR']``.

    Returns
    -------
    None

    Notes
    -----
    The reduction workflow for each observation group consists of:

    1. Extract the science spectrum and associated arc-lamp spectra.
    2. Derive the wavelength solution using the extracted lamp spectra.
    3. Generate a diagnostic comparison between the extracted sky background
       and a reference sky spectrum (if available).
    4. Optionally subtract the estimated background when
       ``SUBTRACT_BKG == 'Y'``.
    5. Optionally perform flux calibration when
       ``FLUX_CALIB == 'Y'``.
    6. Optionally combine all reduced spectra within the group when
       ``SCOMBINE == 'Y'``.

    The function processes every ``ReadyToReduce_group*.txt`` file found in
    the output directory, where each file defines a group of science and
    calibration frames to be reduced.

    See Also
    --------
    extract_obj_lamp
        Extract science and lamp spectra.
    wavelength_calibration
        Derive the wavelength solution from arc-lamp spectra.
    plot_sky
        Generate diagnostic sky comparison plots.
    subtract_background
        Remove the estimated sky background from the extracted spectrum.
    flux_calibration
        Apply the instrument response correction.
    combine_spectra
        Combine multiple reduced spectra into a single spectrum.
    """
    logger = get_logger("spectral")
    dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "ReadyToReduce_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    instrument = instruments[dictkw]
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        logger.info(f"Running for {groupfile}")
        reduced_spectra = []
        for txtline in txtfile_full:
            logger.info(f"Spectral extraction {txtline}")
            optxt_line = extract_obj_lamp(txtline, config, opdir,
                                          instrument['select_trace'])
            wlsolved_fname = wavelength_calibration(optxt_line, config,
                                                    opdir, instrument)
            plot_sky(wlsolved_fname, opdir, instrument['get_stdsky'])
            reduced_spectra.append(wlsolved_fname)
            if config['spectral_extraction']['SUBTRACT_BKG'] == 'Y':
                subtract_background(opdir / wlsolved_fname, config)
            if config['spectral_extraction']['FLUX_CALIB'] == 'Y':
                flux_calibration(opdir / wlsolved_fname, config, instrument)
        if (
                len(reduced_spectra) > 1
                and config['spectral_extraction']['SCOMBINE'] == 'Y'
        ):
            opfilename = Path(reduced_spectra[0]).stem + '.avg.fits'
            opfilename = opdir / opfilename
            reduced_spectra = [opdir / i for i in reduced_spectra]
            logger.info(f"Combining spectra {reduced_spectra}")
            combine_spectra(reduced_spectra,
                            opfilename=opfilename,
                            method=config['inputs']['FRAMECOMBINE'],
                            fluxext=[0, 1, 2],
                            varext=[3, 4, 5],
                            wlext=[6, 6, 6])
            logger.info(f"Output: {opfilename}")
