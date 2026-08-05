#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import ast
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

from .utils import read_txt_file
from .utils import table_to_centroids
from .plottings import imageplot
from .plottings import plot_epsf
from .io_utils import convert_radec

# Detect whether the installed Photutils version supports the
# 'n_brightest' keyword (introduced in Photutils 3.0).
_DAOSTARFINDER_SUPPORTS_N_BRIGHTEST = (
    "n_brightest" in inspect.signature(DAOStarFinder).parameters
)


# -----------------------------
# Locatinig targets
# -----------------------------


def _make_daostarfinder(fwhm, threshold, n_brightest, **kwargs):
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
    imageplot(fname, ext=0, title="Sources found",
              line_profile="aperture", get_target=False,
              centroid_list=centroids,
              save_plot=plot_name,
              show_plot=show_plot,
              aperture_radii=aperture_radii)
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


def aperture_photometry_subrot(config, fname, positions):
    positions = np.array([positions['x_0'],
                          positions['y_0']]).T
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
    elif config['photometry']['ANNUSUS'] == 'EllipticalAnnulus':
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
    # print(phot)
    opfname = save_photometry(
        fname, phot,
        history="Aperture photometry table added on file update.",
        flext=flext
        )

    return opfname


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
        passed to the source finder as initial coordinates. Default is
        ``None``.
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
    psf_model = CircularGaussianSigmaPRF(flux=1,
                                         sigma=fwhm/2.355)
    # print("Select bright targets to generate Effective PSF")
    print("Building effective PSF")
    finder = None
    if star_positions is None:
        print("Finding the sources for ePSF automatically")
        finder = _make_daostarfinder(
            fwhm,
            threshold,
            n_brightest=10,
            xycoords=star_positions,
            min_separation=10)

    psfphot = PSFPhotometry(psf_model, fit_shape,
                            finder=finder,
                            aperture_radius=aperture_radius)
    phot = psfphot(frame,
                   error=err,
                   init_params=star_positions)
    init_flux = np.array(phot['flux_init'])
    x = phot['x_fit']
    y = phot['y_fit']
    mask = init_flux > np.percentile(init_flux, 70)

    epsf_stars_tbl = Table()
    epsf_stars_tbl['x'] = x[mask]
    epsf_stars_tbl['y'] = y[mask]
    nddata = NDData(data=frame,
                    uncertainty=StdDevUncertainty(err))
    epsf_stars = extract_stars(nddata, epsf_stars_tbl,
                               size=cutout_size)

    epsf_builder = EPSFBuilder(oversampling=oversample,
                               smoothing_kernel='quadratic',
                               recentering_maxiters=10,
                               maxiters=10,
                               progress_bar=True)
    epsf, fitted_stars = epsf_builder(epsf_stars)
    plot_epsf(epsf, fitted_stars, plot_fname=plot_fname)
    return epsf


def psf_photometry_subrot(config, fname, positions,
                          plot_dirs="."):
    """
    Perform PSF photometry on sources in a FITS image.

    This function performs point spread function (PSF) photometry using one of
    the supported PSF models (circular Gaussian, elliptical Gaussian, or
    effective PSF). A local background is estimated and subtracted for each
    source before fitting. The resulting photometry table is saved to the
    input FITS file.

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration object containing the photometry settings, input FITS
        extensions, PSF model parameters, fitting parameters, and background
        estimation options.
    fname : str or pathlib.Path
        Path to the input FITS file.
    positions : astropy.table.Table
        Table containing the initial source positions for PSF fitting. The
        table must contain the columns required by
        `photutils.psf.PSFPhotometry`.
    plot_dirs : str or pathlib.Path
        Path to save the plots. Default is ``.``.

    Returns
    -------
    str or pathlib.Path
        Path to the output FITS file containing the PSF photometry results.

    Raises
    ------
    ValueError
        If the aperture radius is greater than or equal to the inner
        background radius, or if the inner background radius is greater than
        or equal to the outer background radius.

    Notes
    -----
    A local background is estimated using
    `photutils.background.MMMBackground` within the annulus defined by
    ``BKGWINDOWS``. The returned PSF fluxes are background-subtracted.
    """
    print("Doing PSF Photometry")
    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']

    fit_shape = ast.literal_eval(config['photometry']['FIT_SHAPE'])
    radius = float(config['photometry']['RADIUS'])
    bkgwindows = ast.literal_eval(config['photometry']['BKGWINDOWS'])

    try:
        varext = int(varext)
    except ValueError:
        print("Using flux array as variance")
        varext = flext

    # Reading data
    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)
    error = np.sqrt(var)

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
                              fwhm=fwhm,
                              threshold=threshold,
                              fit_shape=fit_shape,
                              aperture_radius=radius,
                              plot_fname=plot_fname)

    # background

    if radius >= bkgwindows[0]:
        raise ValueError("RADIUS must be smaller than inner_radius")

    if bkgwindows[0] >= bkgwindows[1]:
        raise ValueError("BKGWINDOWS must be (inner_radius, outer_radius).")

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
    opfname = save_photometry(
        fname, phot,
        history='PSF photometry table added on file update.',
        flext=flext
    )

    return opfname


# -----------------------------
# WCS conversion
# -----------------------------


def save_to_wcs(final_fname):
    opdir = Path(final_fname.parent)
    print(opdir)
    with fits.open(final_fname) as hdul:
        primary_header = hdul[0].header
        w = WCS(primary_header)
        phot_table = Table(hdul['PHOTOMETRY'].data)
        # print(phot_table)
        x = phot_table['x_fit']
        y = phot_table['y_fit']
        # print(x)
        ra, dec = w.wcs_pix2world(x, y, 0)
        # print(ra, dec)
        ra, dec = convert_radec(ra, dec)
        # print(ra, dec)
        colnames = phot_table.colnames
        reordered = Table()
        reordered[colnames[0]] = phot_table[colnames[0]]
        reordered['RA'] = ra
        reordered['Dec'] = dec
        for name in colnames[1:]:
            reordered[name] = phot_table[name]
        # print(reordered)
        out_table_name = final_fname.stem + ".wcs.fits"
        # out_table_path = final_fname.parent
        out_table_name = opdir / out_table_name
        hdu = fits.BinTableHDU(data=reordered, header=primary_header,
                               name='PHOTOMETRY')
        hdul_out = fits.HDUList([fits.PrimaryHDU(header=primary_header), hdu])
        hdul_out.writeto(out_table_name, overwrite=True)
        print("{} saved with WCS coordinates".format(out_table_name))


# -----------------------------
# File Saving
# -----------------------------


def save_photometry(fname, phot_table, history="Photometry table added",
                    flext=0):
    table_hdu = fits.BinTableHDU(phot_table, name="PHOTOMETRY")
    primary_hdu = fits.PrimaryHDU(
        header=fits.getheader(fname, ext=flext)
        )
    hdul = fits.HDUList([primary_hdu, table_hdu])
    opdir = Path(fname.parent)
    opfname = fname.stem + ".phot.fits"
    opfname = opdir / opfname
    hdul.writeto(opfname, overwrite=True)
    return opfname


# -----------------------------
# Centroid
# -----------------------------


def get_centroids(filename, purpose='read', new_centroids=None):
    '''
    if purpose == "read", new centroids will not be taken.
    Otherwise, write the text file with new_centroids
    '''
    if purpose == 'read':
        if not filename.exists():
            return None
        else:
            centroids = read_txt_file(filename)
            # convert y, x to float
            for i, loc in enumerate(centroids):
                centroids[i][0] = float(loc[0])
                centroids[i][1] = float(loc[1])
            # print(centroids)
            if len(centroids) == 0:
                return None
            else:
                return centroids
    elif purpose == 'write':
        targets_txt = open(filename, 'w')
        new_centroids = [list(row)[::-1] for row in new_centroids]
        # print(new_centroids)
        for loc in new_centroids:
            loc_line = " ".join(map(str, loc)) + "\n"
            targets_txt.write(loc_line)
        targets_txt.close()
        print("Updated the selected sources list")


# -----------------------------
# Extraction
# -----------------------------


def photometry_extraction(config, dirname):
    # dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "Readytoextract_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)

    radius = float(config['photometry']['RADIUS'])
    bkgwindows = ast.literal_eval(
        config['photometry']['BKGWINDOWS']
    )

    # Making directory to save plots
    plot_dir = opdir / "Photometry_plots"
    plot_dir.mkdir(exist_ok=True)

    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            frametoextract = txtline[0]
            frametoextract = opdir / frametoextract
            sources_txtfname = opdir / config['photometry']['SOURCELIST']
            editsource = config['photometry']['EDITSOURCE'] == 'YES'
            if config['photometry']['FINDSOURCE'] == 'AUTO':
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
                    centroids = targetfind_manual(
                        frametoextract,
                        centroids_0=table_to_centroids(centroids),
                        aperture_radii=(radius, bkgwindows[0], bkgwindows[1]),
                        plot_dirs=plot_dir
                    )

            elif config['photometry']['FINDSOURCE'] == 'MANUAL':

                centroids_0 = get_centroids(sources_txtfname, purpose='read')
                centroids = targetfind_manual(
                    frametoextract,
                    centroids_0=centroids_0,
                    aperture_radii=(radius, bkgwindows[0], bkgwindows[1]),
                    plot_dirs=plot_dir
                )

            get_centroids(sources_txtfname, purpose='write',
                          new_centroids=centroids)
            # Doing photometry
            if config['photometry']['METHOD'] == 'PSF':
                withphot = psf_photometry_subrot(config, frametoextract,
                                                 positions=centroids,
                                                 plot_dirs=plot_dir)
            elif config['photometry']['METHOD'] == 'Aperture':
                withphot = aperture_photometry_subrot(config, frametoextract,
                                                      positions=centroids)
            print("Photometry data saved to {}".format(withphot))
            save_to_wcs(withphot)
