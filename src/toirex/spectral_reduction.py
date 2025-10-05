#!/usr/bin/env python3

from pathlib import Path
import pprint
import SpectrumExtractor.spectrum_extractor as specextractor

from .instrument import instruments
from .utils import get_pkgpath
from .utils import read_txt_file
from .setups import read_config
from .setups import create_config


def config_for_extraction(data_fname, config,
                          trace_selection):
    """
    Function to create a config file for spectral extraction
    """
    dirname = Path(data_fname.parent)
    # opdir = Path(config['outputs']['OP_DIR']) / dirname
    extractor_fname = config['spectral_extraction']['EXTRACTORCONFIG']
    defaultconfig = False
    # If user does not specify any config file for spectral extraction,
    # uses default config and traces. The traces will be specific for
    # each instrument.
    if (len(extractor_fname) == 0) or (extractor_fname.lower() == 'default'):
        defaultconfig = True
        print('\n \033[1;32m Uses default config file\033[0m' +
              '\033[1;32m (https://github.com/varghesereji/config/' +
              'spectrum_extractor.config)'
              + ' for spectrum extraction' + '\033[0m')
        extractor_fname = get_pkgpath() / 'config/spectrum_extractor.config'
    extraction_config = read_config(extractor_fname)
    tracing_settings = extraction_config['tracing_settings']
    if defaultconfig:
        # Taking trace saved with pipeline
        star_trace, aperture_label, aperturetrace = trace_selection(
            data_fname
        )
        tracing_settings['ContinuumFile'] = str(star_trace)
        tracing_settings['ApertureLabel'] = str(aperture_label)
        tracing_settings['ApertureTraceFilename'] = str(aperturetrace)

    # Setting up aperture windows
    extraction_settings = extraction_config['extraction_settings']
    aperturewindow = config['spectral_extraction']['APERTUREWINDOW']
    bkgwindow = config['spectral_extraction']['BKGWINDOWS']
    extraction_settings['ApertureWindow'] = aperturewindow
    extraction_settings['BkgWindows'] = bkgwindow

    # Creating new configfile
    new_configfname = Path(data_fname).stem + ".config"
    new_configfname = dirname / new_configfname

    create_config(new_configfname, extraction_config)
    return new_configfname


def extract_spectra(txtline, config,
                    opdir, instrument):
    """
    Spectral extraction
    """
    data_fname = opdir / txtline[0]

    extraction_config = config_for_extraction(data_fname,
                                              config,
                                              instrument)
    op_fname = Path(data_fname).stem + ".ms.fits"
    op_fname = Path(opdir) / op_fname

    OutputObjSpec, Avg_XD_shift, PixDomain = specextractor.main(
        [str(data_fname),
         str(extraction_config),
         str(op_fname)]
    )


def spectral_reduction(config, dirname):
    """
    Spectral reduction for each frame.
    """
    dictkw = config['inits']['dictkw']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "ReadyToReduce_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    instrument = instruments[dictkw]
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            extract_spectra(txtline, config, opdir,
                            instrument['select_trace'])

    # traces = instrument['select_trace']
