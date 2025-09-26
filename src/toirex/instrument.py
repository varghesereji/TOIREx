#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
from functools import partial

from .utils import read_fits_header


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
    flat_check = np.array([
        kw.lower() in filename.lower()
        for kw in flats_kws])
    if np.sum(flat_check) > 0:
        flag = "FLAT"
    else:
        flag = "OBJECT"
    flog_list.append(flag)
    return flog_list


#################################
#    Function dictionaries      #
#################################

instruments = {
    'SpecTANSPEC':
    {'sort_filename_key': partial(sort_filename_key, regexp=r'(\d+)\.Z\.fits'),
     'standardise_header': standardise_header_spectanspec,
     'frame_select': frame_select_spectanspec,
     'catalog_flag': catalog_flag_spectanspec,
     'grouping_keys': ['GRATING', 'SLIT',
                       'A_TRGTRA', 'A_TRGTDE'],
     'flat_kw': ['CONT1', 'CONT2', 'SKY'],
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
