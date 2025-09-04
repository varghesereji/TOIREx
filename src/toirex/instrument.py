import re
from pathlib import Path
import numpy as np

from .utils import read_fits_header


'''
Functions for SpecTANSPEC instrument
'''


def sort_filename_key_function_SpecTANSPEC(fname):
    '''
    Function which return the key to sort SpecTANSPEC frames.
    '''
    basename = Path(fname).name
    number = re.search(r'(\d+)\.Z\.fits', basename)
    return int(number.group(1)) if number else -1


def standardiseheader_spectanspec(fname):
    header = read_fits_header(fname)
    header['FNUM'] = int(header['FNUM'])
    return header


def frameselect_decision_spectanspec(fname):
    '''
    This function will decide weather to select a frame or not.
    Since TANSPEC have both image and spectroscopy mode, need to decied weather
    to select certain files or not.
    '''
    header = read_fits_header(fname)
    if (header['NAXIS1'] == 2048) & (header['NAXIS2'] == 2048):
        return True
    else:
        return False


def catalog_flag_tanspec(flog_list):
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
    headers_list = np.array(functions_dict['SpecTANSPEC']['catalog_headers'])

    argon_pos = headers_list == 'ARGONL'
    neon_pos = headers_list == 'NEONL'
    cont1_pos = headers_list == 'CONT1L'
    cont2_pos = headers_list == 'CONT2L'
    calmir_pos = headers_list == 'CALMIR'
    print(flog_list, )
    flog_list_red = np.array(flog_list[1:])
    argon_flag = flog_list_red[argon_pos]
    neon_flag = flog_list_red[neon_pos]
    cont1_flag = flog_list_red[cont1_pos]
    cont2_flag = flog_list_red[cont2_pos]
    calmir_flag = flog_list_red[calmir_pos]

    lamp_flags = [
        argon_flag, neon_flag,
        cont1_flag, cont2_flag,
        ]

    object_flag = 'None'
    if calmir_flag == 'out':
        object_flag = "OBJECT"
    elif (calmir_flag == 'in') and (lamp_flags == ['1', '0',
                                                   '0', '0']):
        object_flag = "ARGON"
    elif (calmir_flag == 'in') and (lamp_flags == ['0', '1',
                                                   '0', '0']):
        object_flag = "NEON"
    elif (calmir_flag == 'in') and (lamp_flags == ['0', '0',
                                                   '1', '0']):
        object_flag = "CONT1"
    elif (calmir_flag == 'in') and (lamp_flags == ['0', '0',
                                                   '0', '1']):
        object_flag = "CONT2"

    flog_list.append(object_flag)
    return flog_list


'''
Functions for TIRSPEC instrument
'''


def sort_filename_key_function_TIRSPEC(fname):
    '''
    Function which return the key to sort SpecTANSPEC frames.
    '''
    basename = Path(fname).name
    number = re.search(r'(\d+)\.Z\.fits', basename)
    return int(number.group(1)) if number else -1


def standardiseheader_tirspec(fname):
    header = read_fits_header(fname)
    fnum = sort_filename_key_function_TIRSPEC(fname)
    header['FNUM'] = fnum
    return header


def frameselect_decision_tirspec(fname):
    '''
    This function will decide weather to select a frame or not.
    Since TANSPEC have both image and spectroscopy mode, need to decied weather
    to select certain files or not.
    '''
    return True


def catalog_flag_tirspec(flog_list):
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
    flog_list.append("OBJECT")
    return flog_list


'''
Function dictionaries
'''
functions_dict = {
    'SpecTANSPEC': {
        'filename_sort_func': sort_filename_key_function_SpecTANSPEC,
        'StandardiseHeader': standardiseheader_spectanspec,
        'frame_select_function': frameselect_decision_spectanspec,
        'catalog_headers': [
            'FNUM', 'A_UTC', 'DATE_OBS', 'CONTGAIN',
            'ITIMEREQ', 'FILTER', 'GRATING', 'SLIT',
            'CALMIR', 'OBJECT',
            'ARGONL', 'NEONL', 'CONT1L', 'CONT2L',
            'A_TRGTRA', 'A_TRGTDE'
                            ],
        'catalog_flag': catalog_flag_tanspec
    },
    'TIRSPEC': {
        'filename_sort_func': sort_filename_key_function_TIRSPEC,
        'StandardiseHeader': standardiseheader_tirspec,
        'frame_select_function': frameselect_decision_tirspec,
        'catalog_headers': [
            'FNUM', 'TIME', 'DATE', 'UPPER',
            'LOWER', 'SLIT', 'CALMIR', 'TARGET',
            'TCSRA', 'TCSDEC'
                            ]
    }
    }
# End
