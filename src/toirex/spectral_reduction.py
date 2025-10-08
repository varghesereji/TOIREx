#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ariastro import combine_process
import SpectrumExtractor.spectrum_extractor as specextractor
from WavelengthCalibrationTool import recalibrate

from astropy.io import fits

from .instrument import instruments
from .utils import get_pkgpath
from .utils import read_txt_file
from .setups import read_config
from .setups import create_config


def config_for_extraction(data_fname, config,
                          trace_selection, for_lamp=None):
    """
    Function to create a config file for spectral extraction.
    To create config for lamp, for_lamp is a dictionary. None else.
    """
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
    if defaultconfig:
        # Taking trace saved with pipeline
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
# Wavelength calibration
# ---------------------------

def wavelength_calibration(txtline, config,
                           opdir, instrument):
    op_fname = opdir / txtline[0]
    arclamp1 = opdir / txtline[1]
    calculate_pixel_offset = instrument['pixel_offset']
    if calculate_pixel_offset is None:
        offset = 0
    else:
        offset = calculate_pixel_offset(arclamp1)
    if len(txtline) > 1:
        comb_lampname = Path(op_fname).stem + "_combarc.fits"
        comb_lampname = opdir / comb_lampname
        lamps_list = [opdir / i for i in txtline[1:]]
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
    # Saving wavelength solution with result
    op_hdul = fits.open(op_fname)
    wlsoln_hdu = fits.ImageHDU(wlsoln, name="Wavelength")
    op_hdul.append(wlsoln_hdu)
    soln_fname = op_fname.stem + ".wlc.fits"
    soln_fname = opdir / soln_fname
    op_hdul.writeto(soln_fname)
    return soln_fname.name

# --------------------- #
#  Spectral Extraction  #
# --------------------- #


def extraction(fname, extraction_config,
               op_fname=None):
    if isinstance(fname, str):
        fname = Path(fname)
    opdir = Path(fname.parent)
    if op_fname is None:
        op_fname = fname.stem + ".ms.fits"
        op_fname = opdir / op_fname

    outputobjspec, avg_xd_shift, pixdomain = specextractor.main(
        [str(fname),
         str(extraction_config),
         str(op_fname)]
    )
    return outputobjspec, avg_xd_shift, pixdomain


def extract_obj_lamp(txtline, config,
                     opdir, instrument):
    """
    Extracting lamp and star spectra.
    """
    data_fname = opdir / txtline[0]

    extraction_config = config_for_extraction(data_fname,
                                              config,
                                              instrument)
    op_fname = Path(data_fname).stem + ".ms.fits"
    op_fname = Path(opdir) / op_fname
    optxtfile_line = [op_fname.name]
    outputobjspec, avg_xd_shift, pixdomain = extraction(data_fname,
                                                        extraction_config)
    refitapertureinxd = [tuple(avg_xd_shift), tuple(pixdomain)]

    # Extracting lamps
    # We want to extract the lamp spectra from the same place where
    # science spectra was extracted.
    lamp_entries = {"ReFitApertureInXD": refitapertureinxd}
    config['spectral_extraction']['EXTRACTORCONFIG'] = str(extraction_config)
    lamp_config = config_for_extraction(data_fname,
                                        config,
                                        instrument,
                                        for_lamp=lamp_entries
                                        )
    lamp_fnames = txtline[1:]

    for n, lamps in enumerate(lamp_fnames):
        lampfile = opdir / lamps
        outlamp_fname = Path(op_fname).stem + "_arc{}.fits".format(n+1)
        optxtfile_line.append(outlamp_fname)
        outlamp_fname = opdir / outlamp_fname
        outputlampspec, avgxdshift, pixdomain = extraction(lampfile,
                                                           lamp_config,
                                                           outlamp_fname)
    return optxtfile_line


# --------------------------------------- #
# Reduction with wavelength calibration   #
# --------------------------------------- #
def spectral_reduction(config, dirname):
    """
    Spectral reduction for each frame.
    """
    dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "ReadyToReduce_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    instrument = instruments[dictkw]
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            optxt_line = extract_obj_lamp(txtline, config, opdir,
                                          instrument['select_trace'])
            wlsolved_fname = wavelength_calibration(optxt_line, config,
                                                    opdir, instrument)

