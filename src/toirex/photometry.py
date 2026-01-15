#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import ast
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.nddata import NDData, StdDevUncertainty

from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPSF
from photutils.psf import GaussianPSF
from photutils.psf import PSFPhotometry
from photutils.psf import IntegratedGaussianPRF
from photutils.psf import extract_stars
from photutils.psf.epsf import EPSFBuilder
from photutils.psf import ImagePSF

from photutils.aperture import CircularAperture
from photutils.aperture import EllipticalAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import aperture_photometry


import matplotlib.pyplot as plt
from .utils import read_txt_file
from .plottings import imageplot
from .io_utils import convert_radec


def targetfind_auto(fname):
    data = fits.getdata(fname, ext=0)
    mean, median, std = sigma_clipped_stats(data)
    daofind = DAOStarFinder(fwhm=6.0, threshold=50,
                            brightest=None, exclude_border=True)
    cl_data = data - median
    plt.figure()
    plt.imshow(cl_data, origin='lower', vmin=0, vmax=mean+std)

    sources = daofind(cl_data)
    # print(sources)
    id_no = sources['id']
    x_pos = sources['xcentroid']
    y_pos = sources['ycentroid']
    for i, index in enumerate(id_no):
        plt.text(x_pos[i], y_pos[i], index)
    plt.show()
    positions = Table()
    positions['x_0'] = x_pos
    positions['y_0'] = y_pos
    return positions


def targetfind_manual(fname, centroids_0):
    centroids = imageplot(fname, ext=0, title="Select sources",
                          line_profile="aperture", get_target=False,
                          centroid_list=centroids_0)
    positions = Table()
    positions['x_0'] = centroids[:, 1]
    positions['y_0'] = centroids[:, 0]
    return positions

# Aperture photometry


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

# PSF photometry


def make_epsf(
        frame,
        err=None,
        star_positions=None,
        cutout_size=25,
        oversample=4,
        normalize=True
):
    """
    Build an effective PSF (ePSF) from a single image frame.

    Parameters
    ----------
    frame : 2D numpy array
        Image containing stars.
    star_positions : list of tuples
        List of (x, y) pixel positions of manually selected stars.
    cutout_size : int
        Size of square cutout (in pixels).
    oversample : int
        Oversampling factor for the ePSF grid.
    normalize : bool
        Normalize each star to unit flux.

    Returns
    -------
    epsf : 2D numpy array
        Oversampled effective PSF.
    """
    psf_model = IntegratedGaussianPRF(flux=200,
                                      sigma=10)
    # print("Select bright targets to generate Effective PSF")
    print("Building effective PSF")
    finder = DAOStarFinder(200,
                           10,
                           xycoords=star_positions,
                           min_separation=30)
    fit_shape = (15, 15)
    psfphot = PSFPhotometry(psf_model, fit_shape,
                            finder=finder,
                            aperture_radius=4)
    phot = psfphot(frame)
    init_flux = np.array(phot['flux_init'])
    x = phot['x_fit']
    y = phot['y_fit']
    mask = init_flux > np.percentile(init_flux, 70)

    epsf_stars_tbl = Table()
    epsf_stars_tbl['x'] = x[mask]
    epsf_stars_tbl['y'] = y[mask]
    nddata = NDData(data=frame,
                    uncertainty=StdDevUncertainty(err))
    epsf_stars = extract_stars(nddata, epsf_stars_tbl, size=25)

    epsf_builder = EPSFBuilder(oversampling=2,
                               smoothing_kernel='quadratic',
                               recentering_maxiters=10,
                               maxiters=10,
                               progress_bar=True)
    epsf, fitted_stars = epsf_builder(epsf_stars)
    return epsf


def psf_photometry_subrot(config, fname, positions):
    print("Doing PSF Photometry")
    fit_shape = (15, 15)
    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']
    try:
        varext = int(varext)
    except ValueError:
        print("Using flux array as variance")
        varext = flext

    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)
    error = np.sqrt(var)

    if config['photometry']['MODEL'] == 'CircularGaussianPSF':
        fwhm = config['photometry']['FWHM']
        print("With CircularGaussianPSF of FWHM", fwhm)
        psf_model = CircularGaussianPSF(flux=1, fwhm=fwhm)

    elif config['photometry']['MODEL'] == 'GaussianPSF':
        psf_fwhm = config['photometry']['PSF_FWHM']
        psf_fwhm = list(float(x) for x in ast.literal_eval(psf_fwhm))
        psf_angle = float(config['photometry']['PSF_ANGLE'])
        print("With GaussianPSF of psf fwhm", psf_fwhm, "and psf angle",
              psf_angle)
        psf_model = GaussianPSF(flux=1,
                                x_fwhm=psf_fwhm[0],
                                y_fwhm=psf_fwhm[1],
                                theta=psf_angle)
    elif config['photometry']['MODEL'] == 'EPSF':
        print("With effective PSF")
        psf_model = make_epsf(data, err=error)

    # print("Saving psf with data frame")
    # size = 25
    # center = (size - 1) / 2
    # y, x = np.mgrid[0:size, 0:size]
    # xg = x - center
    # yg = y - center
    # psf_image = psf_model(xg, yg)
    # psf_image /= np.sum(psf_image)

    # plt.figure()
    # plt.imshow(psf_image)
    # plt.title("PSF")
    # plt.show()

    psfphot = PSFPhotometry(psf_model, fit_shape,
                            aperture_radius=4)

    phot = psfphot(data, error=error, init_params=positions)
    opfname = save_photometry(
        fname, phot,
        history='PSF photometry table added on file update.',
        flext=flext
    )
    return opfname


# WCS conversion


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


# File Saving

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


# Extraction


def photometry_extraction(config, dirname):
    # dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "Readytoextract_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            frametoextract = txtline[0]
            frametoextract = opdir / frametoextract
            sources_txtfname = opdir / config['photometry']['SOURCELIST']
            if config['photometry']['FINDSOURCE'] == 'AUTO':
                centroids = targetfind_auto(frametoextract)
            elif config['photometry']['FINDSOURCE'] == 'MANUAL':
                centroids_0 = get_centroids(sources_txtfname, purpose='read')
                centroids = targetfind_manual(frametoextract,
                                              centroids_0=centroids_0)
            get_centroids(sources_txtfname, purpose='write',
                          new_centroids=centroids)
            # Doing photometry
            if config['photometry']['METHOD'] == 'PSF':
                withphot = psf_photometry_subrot(config, frametoextract,
                                                 positions=centroids)
            elif config['photometry']['METHOD'] == 'Aperture':
                withphot = aperture_photometry_subrot(config, frametoextract,
                                                      positions=centroids)
            print("Photometry data saved to {}".format(withphot))
            save_to_wcs(withphot)
