#!/usr/bin/env python3

import re
from pathlib import Path
try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources

import numpy as np
from functools import partial
from astropy.io import fits

from .utils import read_fits_header
from .utils import read_fits_data
from .utils import get_pkgpath
from .utils import get_pixel_shift
from .setups import read_config


#################################
#      Common functions         #
#################################


def sort_filename_key(fname: str, regexp=r"(\d+)\.Z\.fits"):
    basename = Path(fname).name
    number = re.search(regexp, basename)
    return int(number.group(1)) if number else -1


#################################
#         SpecTANSPEC           #
#################################


def standardise_header_spectanspec(fname: str) -> dict:
    header = read_fits_header(fname)
    header['FNUM'] = int(header['FNUM'])
    return header


def frame_select_spectanspec(fname: str) -> bool:
    '''
    This function will decide weather to select a frame or not.
    Since TANSPEC have both image and spectroscopy mode, need to
    decied weather to select certain files or not.
    '''
    header = read_fits_header(fname)
    return (header['NAXIS1'] == 2048) and (header['NAXIS2'] == 2048)


def catalog_flag_spectanspec(flog_list: list, headers_list: list) -> list:
    '''
    This function is to flag the
    frames as object, argon, neon, cont1
    or cont2. Specific for instrument.
    Input
    ------
    flog_list: catalog entries.
    Return
    ------
    catalog list with flag.
    '''

    headers_list = np.array(headers_list)
    argon_pos = headers_list == 'ARGONL'
    neon_pos = headers_list == 'NEONL'
    cont1_pos = headers_list == 'CONT1L'
    cont2_pos = headers_list == 'CONT2L'
    calmir_pos = headers_list == 'CALMIR'
    flog_list_red = np.array(flog_list[1:])

    argon_flag = flog_list_red[argon_pos].item()
    neon_flag = flog_list_red[neon_pos].item()
    cont1_flag = flog_list_red[cont1_pos].item()
    cont2_flag = flog_list_red[cont2_pos].item()
    calmir_flag = flog_list_red[calmir_pos].item()

    lamp_flags = [
        argon_flag, neon_flag,
        cont1_flag, cont2_flag,
    ]

    object_flag = 'None'
    if calmir_flag == 'out':
        object_flag = "OBJECT"
    elif (calmir_flag == 'in'):
        if (lamp_flags == ['1', '0',
                           '0', '0']):
            object_flag = "ARGON"
        elif (lamp_flags == ['0', '1',
                             '0', '0']):
            object_flag = "NEON"
        elif (lamp_flags == ['0', '0',
                             '1', '0']):
            object_flag = "CONT1"
        elif (lamp_flags == ['0', '0',
                             '0', '1']):
            object_flag = "CONT2"

    flog_list.append(object_flag)
    return flog_list


def makemasterflat_tanspec(normcontdata):
    """
    This function will create a master flat using the continuum flat of each
    night and already generated master flat. This is to take care of noise for
    higher orders (orders 10, 11 nd 12) in XD mode. Basically, we will use
    the master flat to remove noise in higher orders and for the lower orders,
    the pipeline will use the continuum lamp observed in each night for the
    flat correction. In the end it will return the data for the new continuum
    flat.
    """
    mastercontname = 'TANSPEC/CONTLAMPDIR/master-cont1xd_S-1.0.fits'
    continuum_locname = 'TANSPEC/CONTLAMPDIR/ContinuumCutLine.npy'

    # Getting path
    masterconti_path = resources.files("toirex.data") / mastercontname
    continuum_locpath = resources.files("toirex.data") / continuum_locname

    # Loadig file
    mastrrcontidata = read_fits_data(masterconti_path)
    continuum_locdata = np.load(continuum_locpath)

    x_value, y_value = continuum_locdata[:, 0], continuum_locdata[:, 1]
    z = np.polyfit(x_value, y_value, 3)
    p = np.poly1d(z)

    xnewvalue = np.arange(1, normcontdata.shape[0]+1, 1)
    loc_array = p(xnewvalue)

    ynewvalue = np.tile(np.arange(2048), (2048, 1))

    nrows, ncols = ynewvalue.shape
    row, col = np.ogrid[:nrows, :ncols]
    boolmask = row < loc_array

    newflatdata = np.where(boolmask, normcontdata, mastrrcontidata)
    return newflatdata


def masterflat_combination(flat_fname):
    hdul = fits.open(flat_fname, mode='update')
    header = hdul[0].header
    if header['GRATING'] == 'grating1':
        # print("need to create a master flat for ", flat_fname)
        # Updating the flat file
        data = hdul[0].data
        newflatdata = makemasterflat_tanspec(data)
        hdul[0].data = newflatdata
        hdul[0].header.add_history(
            "Created master flat by combine with flats saved with pipeline")
        # Save changes
        hdul.flush()
        # Close the fits file
        hdul.close()
    else:
        hdul.close()
        return flat_fname


def select_trace_spectanspec(
        dataframe,
        instrument_config="config/instrument_templates.config"):
    header = read_fits_header(dataframe)
    pkgpath = get_pkgpath()
    instconfig = pkgpath / instrument_config
    instrument_configs = read_config(instconfig)
    if header["GRATING"] == 'grating1':
        # grating_items = instrument_configs['TANSPEC_XD']
        mode = "XD"
    elif header["GRATING"] == 'grating2':
        # grating_items = instrument_configs['TANSPEC_LR']
        mode = "LR"
    grating_items = instrument_configs['TANSPEC_' + mode]
    star_trace = grating_items['ContinuumFile']
    aperture_label = grating_items['ApertureLabel']
    aperturetrace = grating_items['ApertureTraceFilename']

    star_trace = pkgpath / star_trace
    aperture_label = pkgpath / aperture_label
    aperturetrace = pkgpath / aperturetrace
    return star_trace, aperture_label, aperturetrace


def pixel_offset_spectanspec(lampspectra):
    header = fits.getheader(lampspectra)
    if header['GRATING'] == 'grating2':
        pixeloffset = 0
    elif header['GRATING'] == 'grating1':
        lamp_spec = fits.getdata(lampspectra, ext=0)
        arc_lamp = lamp_spec[7:9].flatten()
        arc_lamp = np.asarray(arc_lamp, dtype=np.float64)
        pkgpath = get_pkgpath()
        template_filename = pkgpath / \
            "data/TANSPEC/XD/pixeloffsettemplate/pixeloffsettemplate_s0.5.npy"
        template = np.load(template_filename)
        pixeloffset = get_pixel_shift(arc_lamp, template)
    return pixeloffset


def get_template_spectanspec(
        lampfname,
        index,
        instrument_config="config/instrument_templates.config"
):
    header = read_fits_header(lampfname)
    pkgpath = get_pkgpath()
    instconfig = pkgpath / instrument_config
    instrument_configs = read_config(instconfig)
    if header["GRATING"] == 'grating1':
        # grating_items = instrument_configs['TANSPEC_XD']
        mode = "XD"
    elif header["GRATING"] == 'grating2':
        # grating_items = instrument_configs['TANSPEC_LR']
        mode = "LR"
    slitwidth = header['SLIT'][2:]
    instrument_specs = instrument_configs['TANSPEC_'+mode]
    temp_path = instrument_specs['LampTrace_dict']
    trace_nameformat = instrument_specs['LampTrace_nameformat']
    if mode == "LR":
        trace_name = trace_nameformat.format("LR")
    elif mode == "XD":
        trace_name = trace_nameformat.format(index)
    template_path = pkgpath / temp_path / slitwidth / trace_name
    template = np.load(template_path)
    return template


def get_response_spectanspec(
        fname,
        instrument_config="config/instrument_templates.config"
):
    header = read_fits_header(fname)
    pkgpath = get_pkgpath()
    instconfig = pkgpath / instrument_config
    instrument_configs = read_config(instconfig)
    if header["GRATING"] == 'grating1':
        # grating_items = instrument_configs['TANSPEC_XD']
        mode = "XD"
    elif header["GRATING"] == 'grating2':
        # grating_items = instrument_configs['TANSPEC_LR']
        mode = "LR"
    instrument_specs = instrument_configs['TANSPEC_'+mode]
    temp_path = pkgpath / instrument_specs['Response']
    return temp_path


def get_stdsky_spectanspec(
        fname,
        instrument_config="config/instrument_templates.config"
):
    header = read_fits_header(fname)
    pkgpath = get_pkgpath()
    instconfig = pkgpath / instrument_config
    instrument_configs = read_config(instconfig)
    if header["GRATING"] == 'grating1':
        # grating_items = instrument_configs['TANSPEC_XD']
        mode = "XD"
    elif header["GRATING"] == 'grating2':
        mode = "LR"

    instrument_specs = instrument_configs['TANSPEC_' + mode]
    sky_fname = pkgpath / instrument_specs['StdSky']
    return sky_fname

#################################
#         TIRSPEC               #
#################################


def standardise_header_tirspec(fname: str) -> dict:
    header = read_fits_header(fname)
    # FNUM not in header. Extracting from filename.
    fnum = instruments['TIRSPEC']['sort_filename_key'](fname)
    header['FNUM'] = fnum
    return header


def frame_select_tirspec(fname: str) -> bool:
    return True


def catalog_flag_tirspec(flog_list: list, headers_list: list) -> list:
    filename = flog_list[0]
    flats_kws = ["flat", "cont"]
    arg_kws = ["ar", "arg"]
    flat_check = np.array([
        kw.lower() in filename.lower()
        for kw in flats_kws])
    arg_check = np.array([
        kw.lower() in filename.lower()
        for kw in arg_kws
        ])
    if np.sum(flat_check) > 0:
        flag = "FLAT"
    elif np.sum(arg_check) > 0:
        flag = "ARGON"
    else:
        flag = "OBJECT"
    flog_list.append(flag)
    return flog_list


def load_badpixelmask_tirspec(
        instrument_config="config/instrument_templates.config"
):
    path = read_config(
        get_pkgpath() / instrument_config
    )[
        'TIRSPEC'
    ][
        'BadPixelMask'
        ]

    mask_path = get_pkgpath() / path
    return mask_path


def select_trace_tirspec(
        dataframe,
        instrument_config="config/instrument_templates.config"):
    header = read_fits_header(dataframe)
    pkgpath = get_pkgpath()
    instconfig = pkgpath / instrument_config
    instrument_configs = read_config(instconfig)
    spec_filter = header["UPPER"]
    grating_items = instrument_configs['TIRSPEC']
    star_trace = grating_items['ContinuumFile'].format(spec_filter,
                                                       spec_filter)
    aperture_label = grating_items['ApertureLabel'].format(spec_filter,
                                                           spec_filter)
    aperturetrace = grating_items['ApertureTraceFilename'].format(spec_filter,
                                                                  spec_filter)

    star_trace = pkgpath / star_trace
    aperture_label = pkgpath / aperture_label
    aperturetrace = pkgpath / aperturetrace
    return star_trace, aperture_label, aperturetrace

# def get_badpixelmask_tirspec()

#################################
#    Function dictionaries      #
#################################

instruments = {
    'SpecTANSPEC':
    {'sort_filename_key': partial(sort_filename_key, regexp=r'(\d+)\.Z\.fits'),
     'standardise_header': standardise_header_spectanspec,
     'frame_select': frame_select_spectanspec,
     'catalog_flag': catalog_flag_spectanspec,
     'masterflat': masterflat_combination,
     'select_trace': select_trace_spectanspec,
     'get_template': get_template_spectanspec,
     'pixel_offset': pixel_offset_spectanspec,
     'get_stdsky': get_stdsky_spectanspec,
     'inst_response': get_response_spectanspec,
     'badpixelmask': None,
     'fname_regexp': r"^(.*?)-\d{5}\.Z\.fits$",
     'grouping_keys': ['GRATING', 'SLIT',
                       'A_TRGTRA', 'A_TRGTDE'],
     'flat_kw': ['CONT1', 'CONT2'],
     'flat_grouping_keys': ['GRATING', 'SLIT'],
     'lamp_kw': ['ARGON', 'NEON'],
     'catalog_headers': [
            'FNUM', 'A_UTC', 'DATE_OBS',
            'ITIMEREQ', 'FILTER', 'GRATING', 'SLIT',
            'CALMIR', 'OBJECT',
            'ARGONL', 'NEONL', 'CONT1L', 'CONT2L',
            'A_TRGTRA', 'A_TRGTDE'
            ]
     },
    'TIRSPEC':
    {'sort_filename_key': partial(sort_filename_key, regexp=r'(\d+)\.Z\.fits'),
     'standardise_header': standardise_header_tirspec,
     'frame_select': frame_select_tirspec,
     'catalog_flag': catalog_flag_tirspec,
     'masterflat': None,
     'select_trace': select_trace_tirspec,
     'pixel_offset': None,
     'get_stdsky': None,
     'inst_response': None,
     'badpixelmask': load_badpixelmask_tirspec,
     'fname_regexp': r"^(.*?)-\d{3}\.Z\.fits$",
     'grouping_keys': ['UPPER', 'LOWER', 'SLIT',
                       'TCSRA', 'TCSDEC'
                       ],
     'flat_kw': ['FLAT'],
     'flat_grouping_keys': ['UPPER', 'LOWER'],
     'lamp_kw': ['ARGON'],
     'catalog_headers': [
            'FNUM', 'TIME', 'DATE', 'UPPER',
            'LOWER', 'SLIT', 'CALMIR', 'TARGET',
            'TCSRA', 'TCSDEC'
        ]
     },
}
# End
