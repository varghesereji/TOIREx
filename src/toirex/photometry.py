#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import ast
import warnings
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.nddata import NDData, StdDevUncertainty

from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPSF
from photutils.psf import CircularGaussianSigmaPRF
from photutils.psf import GaussianPSF
from photutils.psf import PSFPhotometry
from photutils.background import LocalBackground, MMMBackground
# from photutils.psf import IntegratedGaussianPRF
from photutils.psf import extract_stars
try:
    # Support with new versions of photutils (>=2.0.0)
    from photutils.psf import EPSFBuilder
except ModuleNotFoundError:
    from photutils.psf.epsf import EPSFBuilder

from photutils.aperture import CircularAperture
from photutils.aperture import EllipticalAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import aperture_photometry

import inspect

from .setups import get_logger

from .utils import read_txt_file
from .utils import table_to_centroids
from .plottings import imageplot
from .plottings import plot_epsf
from .plottings import save_residualimg
from .io_utils import convert_radec

# Detect whether the installed Photutils version supports the
# 'n_brightest' keyword (introduced in Photutils 3.0).
_DAOSTARFINDER_SUPPORTS_N_BRIGHTEST = (
    "n_brightest" in inspect.signature(DAOStarFinder).parameters
)


# -----------------------------
# Locatinig targets
# -----------------------------


def _make_daostarfinder(fwhm,
                        threshold,
                        n_brightest,
                        **kwargs):

    if _DAOSTARFINDER_SUPPORTS_N_BRIGHTEST:
        return DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            n_brightest=n_brightest,
            **kwargs,
        )

    return DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold,
        brightest=n_brightest,
        **kwargs,
    )


def targetfind_auto(fname,
                    fwhm=7.0,
                    threshold=50,
                    n_brightest=None,
                    xycoords=None,
                    show_plot=True,
                    aperture_radii=(10, 15, 20),
                    plot_dirs="."):
    """
    Automatically detect point sources in an image using DAOStarFinder.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the FITS image.
    fwhm : float, optional
        Full width at half maximum (FWHM) of the point sources in pixels.
        Default is 7.0.
    threshold : float, optional
        Detection threshold above the background in image units.
        Default is 50.
    n_brightest : int or None, optional
        Maximum number of brightest sources to return. If `None`, all
        detected sources are returned.
    xycoords : array-like or None, optional
        Initial source coordinates to use for detection. If `None`,
        sources are detected over the entire image.
    show_plot : bool, optional
        If `True`, display the detected sources overlaid on the image.
        Default is `True`.
    plot_dirs : str or pathlib.Path
        Path to save the plots. Default is ``.``.
    aperture_radii : tuple of float, optional
        Tuple specifying the source aperture radius, background inner
        radius, and background outer radius as
        ``(source_radius, bkg_inner, bkg_outer)``.

    Returns
    -------
    astropy.table.Table
        Table containing the detected source positions with columns
        ``'x_0'`` and ``'y_0'`` which are the x and y coordinates.
    """
    logger = get_logger("photometry")
    logger.info(f"Auto Finding sources in {fname}")
    data = fits.getdata(fname, ext=0)
    _, median, _ = sigma_clipped_stats(data)
    daofind = _make_daostarfinder(
        fwhm=fwhm,
        threshold=threshold,
        n_brightest=n_brightest,
        exclude_border=True,
        xycoords=xycoords
    )
    cl_data = data - median

    sources = daofind(cl_data)
    x_key = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    y_key = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    centroids = table_to_centroids(sources, keys=(y_key, x_key))

    plot_name = fname.with_name(f"{fname.stem}_autoselectedsources.pdf")
    plot_name = Path(plot_dirs) / plot_name.name

    imageplot(
        fname, ext=0,
        title="Sources found",
        line_profile="aperture",
        get_target=False,
        centroid_list=centroids,
        save_plot=plot_name,
        show_plot=show_plot,
        aperture_radii=aperture_radii
    )

    positions = Table()
    positions['x_0'] = sources[x_key]
    positions['y_0'] = sources[y_key]
    return positions


def targetfind_manual(fname,
                      centroids_0,
                      aperture_radii=(10, 15, 20),
                      plot_dirs="."):
    """
    Interactively review and modify source positions in an image.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the FITS image.
    centroids_0 : array-like
        Initial source positions as ``(y, x)`` coordinate pairs. These
        positions are displayed on the image and can be modified
        interactively.
    aperture_radii : tuple of float, optional
        Radii of the circular apertures, in pixels, displayed around each
        source during interactive editing. Default is ``(10, 15, 20)``.
    plot_dirs : str or pathlib.Path
        Path to save the plots. Default is ``.``.

    Returns
    -------
    astropy.table.Table
        Table containing the final source positions after interactive
        editing. The returned table has columns ``'x_0'`` and ``'y_0'``.
    """
    logger = get_logger("photometry")
    logger.info(f"Manual finding sources in {str(fname)}")
    plot_name = fname.with_name(f"{fname.stem}_selectedsources.pdf")
    plot_name = Path(plot_dirs) / plot_name.name
    centroids = imageplot(fname, ext=0, title="Select sources",
                          line_profile="aperture", get_target=False,
                          centroid_list=centroids_0,
                          save_plot=plot_name,
                          aperture_radii=aperture_radii)
    positions = Table()
    positions['x_0'] = centroids[:, 1]
    positions['y_0'] = centroids[:, 0]
    return positions


# -----------------------------
# Aperture photometry
# -----------------------------


def aperture_photometry_subrot(config,
                               fname,
                               positions):
    """
    Perform aperture photometry on the specified sources.

    Aperture and background annulus geometries are selected from the
    photometry configuration. The aperture flux, background level, and
    their associated variances are calculated for each source. The
    resulting photometry table is returned.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the ``photometry`` and
        ``inputs`` sections. The ``photometry`` section must specify
        the aperture type, aperture radius or dimensions, annulus type,
        and background annulus dimensions. The ``inputs`` section must
        specify the FITS extensions containing the flux and variance
        arrays.

    fname : str or pathlib.Path
        Path to the FITS file from which the photometry is extracted.

    positions : astropy.table.Table
        Table containing the initial source positions. The table must
        contain ``x_0`` and ``y_0`` columns giving the source coordinates
        in pixel units.

    Returns
    -------
    astropy.table.QTable
        Table containing the aperture photometry results. The output
        table includes the following columns:

        ``x_fit``, ``y_fit``
            Source coordinates in pixel units.

        ``aperture_sum``, ``aperture_sum_err``
            Total flux within the source aperture and its uncertainty.

        ``bkg``, ``bkg_var``
            Estimated background level per pixel and its variance.

        ``bkg_sum``, ``bkg_var_sum``
            Estimated background contribution within the source aperture
            and its variance.

        ``flux_net``, ``var_net``
            Background-subtracted source flux and its variance.

    Notes
    -----
    The source background is estimated as the median of the pixels
    within the configured background annulus. The background variance
    is estimated from the variance array associated with those pixels.

    The net source flux is calculated as::

        flux_net = aperture_sum - bkg_sum

    and its variance as::

        var_net = aperture_sum_err**2 + bkg_var_sum

    If no valid variance FITS extension is specified, the flux array is
    used as the variance array.

    For elliptical apertures and annuli, the configuration may include
    a rotation angle. If no angle is provided for an elliptical aperture
    or annulus, an angle of zero is used.
    """
    logger = get_logger("photometry")
    positions = np.array([positions['x_0'],
                          positions['y_0']]).T

    logger.info("Doing Aperture photometry")
    logger.info(f"Using {config['photometry']['APERTURE']}")
    logger.info(f"Annulus: {config['photometry']['ANNULUS']}")
    logger.info(f"Radius: {config['photometry']['RADIUS']}")
    logger.info(f"Bkg Annulus: {config['photometry']['BKGWINDOWS']}")

    # Aperture
    if config['photometry']['APERTURE'] == 'CircularAperture':

        radius = float(config['photometry']['RADIUS'])
        apertures = CircularAperture(positions, r=radius)

    elif config['photometry']['APERTURE'] == 'EllipticalAperture':

        aper_qtys = list(float(x)
                         for x in ast.literal_eval(
                                 config['photometry']['RADIUS']
                         )
                         )

        if len(aper_qtys) == 2:

            print("Taking default value for angle, 0")
            theta = 0
            a, b = aper_qtys

        else:

            a, b, theta = aper_qtys
            logger.info(f"Angle {theta}")

        apertures = EllipticalAperture(positions, a=a, b=b, theta=theta)

    # Annulus
    annulus = list(float(x)
                   for x in ast.literal_eval(
                           config['photometry']['BKGWINDOWS']
                   )
                   )

    if config['photometry']['ANNULUS'] == 'CircularAnnulus':

        r_in = annulus[0]
        r_out = annulus[1]
        annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    elif config['photometry']['ANNULUS'] == 'EllipticalAnnulus':

        if len(annulus) == 4:

            theta = 0
            a_in, b_in, a_out, b_out = annulus

        else:

            a_in, b_in, a_out, b_out, theta = annulus

        annulus_apertures = EllipticalAnnulus(positions,
                                              a_in=a_in,
                                              a_out=a_out,
                                              b_in=b_in,
                                              b_out=b_out,
                                              theta=theta)

    annulus_masks = annulus_apertures.to_mask()

    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']

    try:
        varext = int(varext)

    except ValueError:
        print("Using flux array as variance")
        logger.warning("No variance array. Using flux array as variance")
        varext = flext

    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)

    # Aperture
    phot = aperture_photometry(data, apertures,
                               error=np.sqrt(var))

    # Background
    bkg_median = []
    bkg_var = []

    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_var = mask.multiply(var)
        annulus_data_1d = annulus_data[mask.data > 0]
        annulus_var_1d = annulus_var[mask.data > 0]
        median_sigclip = np.median(annulus_data_1d)
        var_clip = np.sum(annulus_var_1d) / np.size(annulus_var_1d) ** 2
        bkg_median.append(median_sigclip)
        bkg_var.append(var_clip)

    bkg_median = np.array(bkg_median)
    bkg_var = np.array(bkg_var)

    phot['bkg'] = bkg_median
    phot['bkg_var'] = bkg_var
    phot['bkg_sum'] = bkg_median * apertures.area
    phot['bkg_var_sum'] = bkg_var * apertures.area
    phot['flux_net'] = phot['aperture_sum'] - phot['bkg_sum']
    phot['var_net'] = phot['aperture_sum_err'] ** 2 + phot['bkg_var_sum']
    phot.rename_column('xcenter', 'x_fit')
    phot.rename_column('ycenter', 'y_fit')

    return phot


# -----------------------------
# PSF photometry
# -----------------------------


def make_epsf(
        frame,
        err=None,
        star_positions=None,
        aperture_radius=4,
        fwhm=7.0,
        threshold=50,
        cutout_size=25,
        fit_shape=(15, 15),
        oversample=4,
        plot_fname="epsf_plot.pdf"
):
    """
    Build an effective point spread function (ePSF) from bright stars in an
    image.

    Bright stars are first refined using PSF photometry and then used to
    construct an oversampled ePSF with ``EPSFBuilder``. The resulting ePSF
    can be used as the PSF model for subsequent PSF photometry.

    Parameters
    ----------
    frame : ndarray
        Two-dimensional image array containing the stellar sources.
    err : ndarray, optional
        Two-dimensional array containing the 1-sigma uncertainty for each
        pixel. If provided, it is used when extracting stellar cutouts.
        Default is ``None``.
    star_positions : array-like, optional
        Initial estimates of the stellar positions. If provided, these are
        passed directly to `PSFPhotometry` as the initial source positions.
        If `None`, sources are detected automatically using DAOStarFinder.
    aperture_radius : float, optional
        Radius of the circular aperture, in pixels, used to estimate the
        initial stellar fluxes during PSF photometry. Default is ``4``.
    fwhm : float, optional
        Approximate full width at half maximum (FWHM) of the stellar PSF, in
        pixels. This is used to define both the initial Gaussian PSF model
        and the source finder. Default is ``7.0``.
    threshold : float, optional
        Detection threshold above the background, in image units, used by the
        source finder. Default is ``50``.
    cutout_size : int, optional
        Size of the square cutout, in pixels, extracted around each selected
        star for ePSF construction. Default is ``25``.
    fit_shape : tuple of int, optional
        Shape of the fitting region used for PSF photometry, given as
        ``(ny, nx)``. Default is ``(15, 15)``.
    oversample : int, optional
        Oversampling factor of the output ePSF. Default is ``4``.
    plot_fname : str or pathlib.Path, optional
        Filename of the output diagnostic plot showing the constructed ePSF.
        Default is ``"epsf_plot.pdf"``.

    Returns
    -------
    photutils.psf.ImagePSF
        The constructed oversampled effective point spread function.
    """
    logger = get_logger("photometry")
    logger.info("Building ePSF")

    psf_model = CircularGaussianSigmaPRF(flux=1,
                                         sigma=fwhm/2.355)

    print("Building effective PSF")

    finder = None

    if star_positions is None:

        print("Finding the sources for ePSF automatically")

        finder = _make_daostarfinder(
            fwhm,
            threshold,
            n_brightest=10,
            min_separation=10)

    psfphot = PSFPhotometry(psf_model, fit_shape,
                            finder=finder,
                            aperture_radius=aperture_radius)

    phot = psfphot(frame,
                   error=err,
                   init_params=star_positions)

    good = phot["flags"] == 0
    phot = phot[good]

    if len(phot) == 0:
        raise ValueError("No stars available for ePSF construction.")

    init_flux = np.asarray(phot['flux_init'])
    x = phot['x_fit']
    y = phot['y_fit']
    mask = init_flux > np.percentile(init_flux, 90)

    epsf_stars_tbl = Table()
    epsf_stars_tbl['x'] = x[mask]
    epsf_stars_tbl['y'] = y[mask]

    nddata = NDData(data=frame,
                    uncertainty=StdDevUncertainty(err))

    epsf_stars = extract_stars(nddata, epsf_stars_tbl,
                               size=cutout_size)
    if len(epsf_stars) < 5:
        msg = (
            f"Only {len(epsf_stars)} bright star(s) were selected for ePSF "
            "construction. The resulting ePSF may be unreliable. "
            "Consider using a Gaussian PSF "
            "model or selecting more bright, isolated stars."
            )

        warnings.warn(
            msg,
            UserWarning,
            stacklevel=2,
        )
        logger.warning(msg)

    logger.info("Building ePSF")

    epsf_builder = EPSFBuilder(oversampling=oversample,
                               smoothing_kernel='quadratic',
                               recentering_maxiters=10,
                               maxiters=10,
                               progress_bar=True)

    epsf, fitted_stars = epsf_builder(epsf_stars)

    logger.info(f"ePSF plot saved as {plot_fname}")

    plot_epsf(epsf, fitted_stars, plot_fname=plot_fname)

    return epsf


def psf_photometry_subrot(config,
                          fname,
                          positions,
                          plot_dirs="."):
    """
    Perform PSF photometry on sources in a FITS image.

    The function performs point spread function (PSF) photometry using one
    of the configured PSF models: circular Gaussian, elliptical Gaussian,
    or effective PSF (ePSF). A local background is estimated for each
    source using an annular region and ``MMMBackground``. The fitted source
    parameters and fluxes are returned as an Astropy table.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the ``photometry`` and
        ``inputs`` sections. The ``photometry`` section must specify the
        PSF model, fitting shape, aperture radius, background annulus,
        and model-specific parameters. The ``inputs`` section must specify
        the FITS extensions containing the flux and variance arrays.

    fname : str or pathlib.Path
        Path to the input FITS file containing the image data.

    positions : astropy.table.Table
        Table containing the initial source positions used to initialize
        the PSF fitting. The table must contain the columns required by
        ``photutils.psf.PSFPhotometry`` for initial source parameters,
        including the source coordinates.

    plot_dirs : str or pathlib.Path, optional
        Directory in which diagnostic plots are saved. For the ePSF model,
        the generated ePSF diagnostic plot is saved here. A PSF residual
        image is also saved after the photometry is performed.
        Default is ``.``.

    Returns
    -------
    astropy.table.Table
        Table containing the PSF photometry results returned by
        ``photutils.psf.PSFPhotometry``. The table contains the fitted
        source parameters, including the fitted source coordinates and
        PSF fluxes and associated uncertainties.

    Raises
    ------
    ValueError
        If the source aperture radius is greater than or equal to the inner
        background radius, or if the inner background radius is greater
        than or equal to the outer background radius.

    Notes
    -----
    The variance array is read from the FITS extension specified by
    ``VAREXT``. If a valid variance extension is not specified, the flux
    array is used as the variance array.

    The local background is estimated using
    ``photutils.background.MMMBackground`` within the annulus specified
    by ``BKGWINDOWS``. The estimated background is supplied to
    ``PSFPhotometry`` during the fitting.

    Supported PSF models are:

    ``CircularGaussianPSF``
        Circular Gaussian PSF with the configured FWHM.

    ``GaussianPSF``
        Elliptical Gaussian PSF with configurable x and y FWHM values
        and rotation angle.

    ``EPSF``
        Effective PSF constructed from the image using ``make_epsf``.
        The ePSF construction also produces a diagnostic plot.

    A residual image showing the difference between the input image and
    the fitted PSF model is saved to ``plot_dirs`` after the photometry
    is completed.
    """
    logger = get_logger("photometry")

    print("Doing PSF Photometry")

    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']

    fit_shape = ast.literal_eval(config['photometry']['FIT_SHAPE'])
    radius = float(config['photometry']['RADIUS'])
    bkgwindows = ast.literal_eval(config['photometry']['BKGWINDOWS'])

    logger.info(f"Radius: {config['photometry']['RADIUS']}")
    logger.info(f"Bkg Annulus: {config['photometry']['BKGWINDOWS']}")
    logger.info(f"fit_shape: {fit_shape}")

    try:
        varext = int(varext)

    except ValueError:
        print("Using flux array as variance")
        logger.warning("No variance array. Using flux array as variance")

        varext = flext

    # Reading data
    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)
    error = np.sqrt(var)

    logger.info(f"PSF model: {config['photometry']['MODEL']}")

    # Making PSF
    if config['photometry']['MODEL'] == 'CircularGaussianPSF':
        fwhm = float(config['photometry']['FWHM'])
        print("With CircularGaussianPSF of FWHM", fwhm)
        psf_model = CircularGaussianPSF(flux=1, fwhm=fwhm)

    elif config['photometry']['MODEL'] == 'GaussianPSF':
        psf_fwhm = config['photometry']['PSF_FWHM']
        psf_fwhm = tuple(float(x) for x in ast.literal_eval(psf_fwhm))
        psf_angle = float(config['photometry']['PSF_ANGLE'])
        print("With GaussianPSF of psf fwhm", psf_fwhm, "and psf angle",
              psf_angle)
        psf_model = GaussianPSF(flux=1,
                                x_fwhm=psf_fwhm[0],
                                y_fwhm=psf_fwhm[1],
                                theta=psf_angle)

    elif config['photometry']['MODEL'] == 'EPSF':
        print("With effective PSF")
        fwhm = float(config['photometry']['FWHM'])
        threshold = float(config['photometry']['THRESHOLD'])
        plot_fname = fname.with_name(f"{fname.stem}_epsf.pdf")
        plot_fname = Path(plot_dirs) / plot_fname.name
        psf_model = make_epsf(data, err=error,
                              star_positions=positions,
                              fwhm=fwhm,
                              threshold=threshold,
                              fit_shape=fit_shape,
                              aperture_radius=radius,
                              plot_fname=plot_fname)

    # background

    if radius >= bkgwindows[0]:
        msg = "RADIUS must be smaller than inner_radius"
        logger.error(msg)
        raise ValueError(msg)

    if bkgwindows[0] >= bkgwindows[1]:
        msg = "BKGWINDOWS must be (inner_radius, outer_radius)."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Using MMMBackground")
    bkgstat = MMMBackground()
    local_bkg_estimator = LocalBackground(bkgwindows[0],
                                          bkgwindows[1],
                                          bkg_estimator=bkgstat)

    # PSF photometry

    psfphot = PSFPhotometry(psf_model, fit_shape,
                            local_bkg_estimator=local_bkg_estimator,
                            aperture_radius=radius,
                            progress_bar=True)

    phot = psfphot(data, error=error, init_params=positions)
    resplot_fname = fname.with_name(f"{fname.stem}_psfresidue.pdf")
    resplot_fname = Path(plot_dirs) / resplot_fname.name

    save_residualimg(data,
                     psfphot.make_residual_image(data),
                     fname=resplot_fname,
                     show_plot=True)

    return phot


# -----------------------------
# WCS conversion
# -----------------------------


def save_to_wcs(final_fname):
    """
    Add WCS coordinates to the photometry table and save the result.

    The fitted pixel coordinates (``x_fit`` and ``y_fit``) from the
    ``PHOTOMETRY`` extension are converted to celestial coordinates
    (RA and Dec) using the WCS information in the primary HDU header.
    The resulting RA and Dec columns are added to the photometry table,
    and the updated table is saved to a new FITS file with a ``.wcs.fits``
    suffix.

    A history entry is also added to the primary FITS header to record
    the addition of the WCS coordinates.

    Parameters
    ----------
    final_fname : pathlib.Path
        Path to the input photometry FITS file.

    Returns
    -------
    None
        The WCS-coordinate photometry table is written to a new FITS file.
    """
    logger = get_logger("photometry")

    opdir = Path(final_fname.parent)

    with fits.open(final_fname) as hdul:
        primary_header = hdul[0].header

        primary_header.add_history(
            "RA and Dec coordinates added to photometry table"
            )

        w = WCS(primary_header)

        if not w.has_celestial:
            logger.warning(
                "No celestial WCS information found in %s."
                "Skipping WCS correction.",
                final_fname,
                )
            return

        phot_table = Table(hdul['PHOTOMETRY'].data)

        x = phot_table['x_fit']
        y = phot_table['y_fit']

        ra, dec = w.wcs_pix2world(x, y, 0)
        ra, dec = convert_radec(ra, dec)

        colnames = phot_table.colnames

        reordered = Table()

        reordered[colnames[0]] = phot_table[colnames[0]]

        reordered['RA'] = ra
        reordered['Dec'] = dec

        for name in colnames[1:]:
            reordered[name] = phot_table[name]

        out_table_name = final_fname.stem + ".wcs.fits"
        out_table_name = opdir / out_table_name

        hdu = fits.BinTableHDU(data=reordered, header=primary_header,
                               name='PHOTOMETRY')
        hdul_out = fits.HDUList([fits.PrimaryHDU(header=primary_header), hdu])

        hdul_out.writeto(out_table_name, overwrite=True)

    print("{} saved with WCS coordinates".format(out_table_name))
    logger.info("Photometry data saved with WCS coordinates")

# -----------------------------
# File Saving
# -----------------------------


def save_photometry(fname,
                    phot_table,
                    output_filename=None,
                    history="Photometry table added",
                    save_magnitude=True,
                    flext=0):
    """
    Save a photometry table to a new FITS file.

    The photometry table is stored in a ``PHOTOMETRY`` binary table
    extension. The header from the specified input extension is copied
    to the primary HDU of the output file, with a history entry added
    to record the photometry operation.

    Parameters
    ----------
    fname : pathlib.Path
        Path to the input FITS file.

    phot_table : astropy.table.Table
        Photometry table to be saved in the output FITS file.

    output_fname : str or pathlib.Path, optional
        Output filename for the photometry FITS file. If ``None``, the
        output filename is generated from ``fname`` by replacing its
        extension with ``".phot.fits"``.

    history : str, optional
        History entry to add to the output FITS header.
        Default is ``"Photometry table added"``.

    flext : int, optional
        FITS extension from which the input header is obtained.
        Default is ``0``.

    Returns
    -------
    pathlib.Path
        Path to the output photometry FITS file.
    """
    logger = get_logger("photometry")

    header = fits.getheader(fname, ext=flext)

    # calculating snr
    calculate_snr(phot_table)
    header.add_history("Calculated signal-to-noise ratio of each source")

    if save_magnitude:
        calculate_magnitude(phot_table)
        header.add_history("Calculated instrument magnitude")

    table_hdu = fits.BinTableHDU(phot_table, name="PHOTOMETRY")

    header.add_history(history)

    primary_hdu = fits.PrimaryHDU(
        header=header
        )

    hdul = fits.HDUList([primary_hdu, table_hdu])

    if output_filename is None:
        opdir = Path(fname.parent)
        output_filename = fname.stem + ".phot.fits"
        output_filename = opdir / output_filename

    hdul.writeto(output_filename, overwrite=True)

    logger.info(f"Photometry file saved as {output_filename}")

    return output_filename


# -----------------------------
# Centroid
# -----------------------------


def get_centroids(filename,
                  purpose='read',
                  new_centroids=None):
    """
    if purpose == "read", new centroids will not be taken.
    Otherwise, write the text file with new_centroids
    """
    if purpose == 'read':

        if not filename.exists():
            return None

        else:
            centroids = read_txt_file(filename)

            # convert y, x to float
            for i, loc in enumerate(centroids):
                centroids[i][0] = float(loc[0])
                centroids[i][1] = float(loc[1])

            if len(centroids) == 0:
                return None

            else:
                return centroids

    elif purpose == 'write':

        targets_txt = open(filename, 'w')
        new_centroids = [list(row)[::-1] for row in new_centroids]

        for loc in new_centroids:
            loc_line = " ".join(map(str, loc)) + "\n"
            targets_txt.write(loc_line)

        targets_txt.close()
        print("Updated the selected sources list")


# -----------------------------
# SNR
# -----------------------------

def calculate_snr(phot_table):
    """
    Calculate the signal-to-noise ratio for each detected source.

    For aperture photometry, the flux uncertainty is calculated as the
    square root of ``var_net``. For PSF photometry, the existing
    ``flux_err`` column is used.

    Parameters
    ----------
    phot_table : astropy.table.Table
        Photometry table containing either ``flux_net`` and ``var_net``
        for aperture photometry, or ``flux_fit`` and ``flux_err`` for
        PSF photometry.

    Returns
    -------
    astropy.table.Table
        The input photometry table with an additional ``snr`` column.
    """
    logger = get_logger("photometry")

    logger.info("Calculating signal-to-noise ratio")
    print("Calculating SNR")

    # Case in aperture photometry
    if "flux_net" in phot_table.colnames:

        flux = phot_table['flux_net']
        flux_err = np.sqrt(phot_table['var_net'])

    # Case in PSF photometry
    elif "flux_fit" in phot_table.colnames:
        flux = phot_table['flux_fit']
        flux_err = phot_table['flux_err']

    # If not both of these cases
    else:
        phot_cols = phot_table.colnames
        errormsg = "Could not identify photometry type from photometry table;"
        errormsg += " expected aperture columns ('flux_net', 'var_net') or PSF"
        errormsg += " columns ('flux_fit', 'flux_err'), but got columns: "
        errormsg += f"{phot_cols}"

        logger.error(errormsg)
        raise ValueError(
            errormsg
        )

    # snr_calculation
    phot_table['snr'] = flux / flux_err

    return phot_table


# -----------------------------
# Magnitude
# -----------------------------
def calculate_magnitude(phot_table):
    """
    Calculate instrumental magnitudes and their uncertainties.

    The function supports both aperture and PSF photometry. For aperture
    photometry, the net flux is taken from ``flux_net`` and its uncertainty
    is calculated from the variance ``var_net``. For PSF photometry, the
    fitted flux and its uncertainty are taken from ``flux_fit`` and
    ``flux_err``, respectively.

    The instrumental magnitude is calculated as

    .. math::

        m = -2.5 \\log_{10}(F),

    and its uncertainty is propagated from the flux uncertainty as

    .. math::

        \\sigma_m = \\frac{2.5}{\\ln(10)}
        \\frac{\\sigma_F}{F}.

    Parameters
    ----------
    phot_table : astropy.table.Table
        Photometry table containing either the aperture photometry columns
        ``flux_net`` and ``var_net``, or the PSF photometry columns
        ``flux_fit`` and ``flux_err``.

    Returns
    -------
    astropy.table.Table
        The input photometry table with two additional columns,
        ``mag`` and ``mag_err``, containing the instrumental magnitude
        and its uncertainty.

    Raises
    ------
    ValueError
        If the photometry table does not contain the expected columns for
        either aperture or PSF photometry.
    """
    logger = get_logger("photometry")

    logger.info("Calculating instrument magnitude")
    print("Calculating instrument magnitudes")

    # Case in aperture photometry
    if "flux_net" in phot_table.colnames:

        flux = phot_table['flux_net']
        flux_err = np.sqrt(phot_table['var_net'])

    # Case in PSF photometry
    elif "flux_fit" in phot_table.colnames:
        flux = phot_table['flux_fit']
        flux_err = phot_table['flux_err']

    # If not both of these cases
    else:
        phot_cols = phot_table.colnames
        errormsg = "Could not identify photometry type from photometry table;"
        errormsg += " expected aperture columns ('flux_net', 'var_net') or PSF"
        errormsg += " columns ('flux_fit', 'flux_err'), but got columns: "
        errormsg += f"{phot_cols}"

        logger.error(errormsg)
        raise ValueError(
            errormsg
        )

    # Magnitude calculation
    phot_table["mag"] = -2.5 * np.log10(flux)
    phot_table["mag_err"] = (2.5 / np.log(10)) * (flux_err / flux)

    return phot_table


# -----------------------------
# Source finding
# -----------------------------

def find_sources(
        config,
        frametoextract,
        plot_dir
):
    """
    Find source positions in an astronomical image.

    Sources can be identified automatically or manually according to the
    ``FINDSOURCE`` configuration option. In automatic mode, sources are
    detected using :func:`targetfind_auto`. If source editing is enabled,
    the automatically detected sources are subsequently passed to
    :func:`targetfind_manual` for interactive adjustment.

    In manual mode, initial source positions are read from the configured
    source list and passed to :func:`targetfind_manual`.

    The final source positions are written to the configured source list
    and returned.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the photometry settings,
        including source-finding method, aperture radius, background
        windows, source list, FWHM, and detection threshold.

    frametoextract : str or pathlib.Path
        Path to the frame in which sources are to be identified.

    plot_dir : str or pathlib.Path
        Directory where source-finding plots are saved.

    Returns
    -------
    astropy.table.Table
        Table containing the final source positions.

    Raises
    ------
    KeyError
        If required photometry configuration parameters are missing.

    ValueError
        If a numerical configuration parameter cannot be converted to
        the required type.
    """

    logger = get_logger("photometry")
    opdir = Path(frametoextract).parent

    radius = float(config['photometry']['RADIUS'])
    bkgwindows = ast.literal_eval(
        config['photometry']['BKGWINDOWS']
    )

    sources_txtfname = opdir / config['photometry']['SOURCELIST']
    editsource = config['photometry']['EDITSOURCE'] == 'YES'
    if config['photometry']['FINDSOURCE'] == 'AUTO':
        logger.info("Finding source AUTO")
        fwhm = float(config['photometry']['FWHM'])
        threshold = float(config['photometry']['THRESHOLD'])
        centroids = targetfind_auto(
            frametoextract,
            fwhm=fwhm,
            threshold=threshold,
            show_plot=not editsource,
            plot_dirs=plot_dir,
            aperture_radii=(radius, bkgwindows[0], bkgwindows[1]),
        )

        if editsource:
            logger.info("Editing the sources found")
            centroids = targetfind_manual(
                frametoextract,
                centroids_0=table_to_centroids(centroids),
                aperture_radii=(radius, bkgwindows[0], bkgwindows[1]),
                plot_dirs=plot_dir
            )

    elif config['photometry']['FINDSOURCE'] == 'MANUAL':
        logger.info("Finding source MANUAL")
        centroids_0 = get_centroids(sources_txtfname, purpose='read')
        centroids = targetfind_manual(
            frametoextract,
            centroids_0=centroids_0,
            aperture_radii=(radius, bkgwindows[0], bkgwindows[1]),
            plot_dirs=plot_dir
        )

    get_centroids(sources_txtfname, purpose='write',
                  new_centroids=centroids)

    return centroids


# -----------------------------
# Extraction
# -----------------------------

def extract_photometry(
        config,
        frametoextract,
        centroids,
        plot_dir,
        output_filename=None
):
    """
    Extract and save photometry using the configured photometry method.

    The photometry method is selected from the ``METHOD`` entry in the
    ``photometry`` configuration. PSF photometry is performed using
    :func:`psf_photometry_subrot`, while aperture photometry is performed
    using :func:`aperture_photometry_subrot`. The resulting photometry
    table is then saved using :func:`save_photometry`.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the ``photometry`` and
        ``inputs`` sections. The ``photometry`` section must specify
        ``METHOD`` and ``SAVE_MAGNITUDE``. The ``inputs`` section must
        specify the flux FITS extension through ``FLUXEXT``.

    frametoextract : str or pathlib.Path
        Path to the FITS frame from which photometry is to be extracted.

    centroids : astropy.table.Table or array-like
        Initial source positions to be used for photometry. For PSF
        photometry, the positions are passed to
        :func:`psf_photometry_subrot`. For aperture photometry, they are
        passed to :func:`aperture_photometry_subrot`.

    plot_dir : str or pathlib.Path
        Directory in which diagnostic plots generated during PSF
        photometry are saved. This argument is ignored for aperture
        photometry.

    output_fname : str or pathlib.Path, optional
        Output filename for the photometry FITS file. If ``None``, the
        output filename is generated from ``fname`` by replacing its
        extension with ``".phot.fits"``.

    Returns
    -------
    str or pathlib.Path
        Path to the FITS file containing the saved photometry table.

    Raises
    ------
    KeyError
        If required entries such as ``photometry``, ``METHOD``,
        ``SAVE_MAGNITUDE``, ``inputs``, or ``FLUXEXT`` are missing from
        ``config``.

    ValueError
        If ``METHOD`` is not ``'PSF'`` or ``'Aperture'``.

    Notes
    -----
    If ``SAVE_MAGNITUDE`` is set to ``'Y'``, magnitude and magnitude
    uncertainty are calculated and included in the saved photometry
    table by :func:`save_photometry`.

    The photometry method-specific routine returns the photometry table,
    which is then passed to :func:`save_photometry` along with the input
    frame and flux extension information.
    """
    logger = get_logger('photometry')

    # Doing photometry
    if config['photometry']['METHOD'] == 'PSF':
        logger.info("Doing PSF Photometry")
        phot_table = psf_photometry_subrot(config, frametoextract,
                                           positions=centroids,
                                           plot_dirs=plot_dir)
        logger.info("PSF Photometry DONE")
        history = "PSF photometry table added on file update."

    elif config['photometry']['METHOD'] == 'Aperture':
        phot_table = aperture_photometry_subrot(config, frametoextract,
                                                positions=centroids)
        logger.info("Aperture Photometry DONE")
        history = "Aperture photometry table added on file update."

    save_magnitude = config['photometry']['SAVE_MAGNITUDE'] == 'Y'
    flext = int(config['inputs']['FLUXEXT'])

    withphot = save_photometry(
        frametoextract,
        phot_table,
        output_filename=output_filename,
        history=history,
        save_magnitude=save_magnitude,
        flext=flext
        )

    print("Photometry data saved to {}".format(withphot))
    logger.info(f"Output saved as {withphot}")

    return withphot


# -----------------------------
# Processes of photometry
# -----------------------------


def phot_process(config,
                 frametoextract,
                 opdir=None,
                 output_filename=None):
    """
    Perform the complete photometry processing for a single frame.

    The frame path is resolved relative to ``opdir`` when provided.
    A directory for photometry plots is created within ``opdir``.
    Sources are then identified, photometry is extracted for the
    detected sources, and WCS coordinates are added to the resulting
    photometry table.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the photometry settings.

    frametoextract : str or pathlib.Path
        Name or path of the frame to be processed. If ``opdir`` is
        provided, the frame is interpreted relative to that directory.

    opdir : str or pathlib.Path, optional
        Output directory containing the frame and where photometry
        products and plots will be saved. If ``None``, the parent
        directory of ``frametoextract`` is used.

    output_fname : str or pathlib.Path, optional
        Output filename for the photometry FITS file. If ``None``, the
        output filename is generated from ``fname`` by replacing its
        extension with ``".phot.fits"``.

    Returns
    -------
    None
        The processed photometry is saved to disk, and no value is
        returned.
    """

    logger = get_logger("photometry")

    frametoextract = Path(frametoextract)
    if opdir is None:
        opdir = frametoextract.parent

    else:
        opdir = Path(opdir)

    frametoextract = opdir / frametoextract

    # Making directory to save plots
    plot_dir = opdir / "Photometry_plots"
    plot_dir.mkdir(exist_ok=True)

    centroids = find_sources(
        config,
        frametoextract,
        plot_dir
        )

    withphot = extract_photometry(
        config,
        frametoextract,
        centroids,
        plot_dir,
        output_filename=output_filename
        )

    logger.info(f"WCS correction on {withphot}")
    save_to_wcs(withphot)


def photometry_extraction(config,
                          dirname):
    """
    Perform photometry extraction for all frames in a directory.

    Searches the specified output directory for group files matching
    ``Readytoextract_group*.txt``. Each group file contains the names of
    frames to be processed. The frames are read from the group files and
    passed to :func:`phot_process` for source detection, photometry
    extraction, and WCS correction.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the output directory under
        ``config['outputs']['OP_DIR']`` and the photometry settings.

    dirname : str
        Name of the directory within the configured output directory
        containing the group files and frames to be processed.

    Returns
    -------
    None
        This function performs photometry extraction and does not return
        a value.
    """

    logger = get_logger("photometry")
    logger.info("Doing photometry")

    opdir = Path(config['outputs']['OP_DIR']) / dirname

    reduce_txtfname = "Readytoextract_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)

    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)

        logger.info(f"Running for files in {groupfile}")

        for txtline in txtfile_full:
            frametoextract = txtline[0]

            phot_process(config, frametoextract,
                         opdir)


# End
