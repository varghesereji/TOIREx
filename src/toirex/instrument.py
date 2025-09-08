#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

from .utils import read_fits_header



class Instrument(ABC):
    """ Base class for instruments"""

    def ___init__(self, name):
        self.name = name

    @abstractmethod
    def sort_filename_key(self, fname: str) -> int:
        """ Return a key for sorting instrument files"""
        pass

    @abstractmethod
    def standardise_header(self, fname: str) -> dict:
        """ Standardise the FITS header. """
        pass

    @abstractmethod
    def frame_select(self, fname:str) -> bool:
        """Decide weather to select a frame or not."""
        pass

    @abstractmethod
    def catalog_flag(self, flog_list: list) -> list:
        """Return a flagged catalog entry"""
        pass

    @property
    @abstractmethod
    def catalog_headers(self) -> list:
        """ List of catalog headers for this instrument."""
        pass

    @property
    def grouping_keys(self) -> list:
        """Keys used for grouping, optional"""
        return []

    @property
    def flat_grouping_keys(self) -> list:
        return []

    @property
    def flat_kw(self) -> list:
        return []


class SpecTANSPEC(Instrument):
    name = "SpecTANSPEC"

    def __init__(self):
        pass

    def sort_filename_key(self, fname: str) -> int:
        basename = Path(fname).name
        number = re.search(r'(\d+)\.Z\.fits', basename)
        return int(number.group(1)) if number else -1

    def standardise_header(self, fname: str) -> dict:
        header = read_fits_header(fname)
        header['FNUM'] = int(header['FNUM'])
        return header

    def frame_select(self, fname: str) -> bool:
        '''
        This function will decide weather to select a frame or not.
        Since TANSPEC have both image and spectroscopy mode, need to
        decied weather to select certain files or not.
        '''
        header = read_fits_header(fname)
        return (header['NAXIS1'] == 2048) and (header['NAXIS2'] == 2048)

    def catalog_flag(self, flog_list: list) -> list:
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
        headers_list = self.catalog_headers

        headers_list = np.array(headers_list)
        argon_pos = headers_list == 'ARGONL'
        neon_pos = headers_list == 'NEONL'
        cont1_pos = headers_list == 'CONT1L'
        cont2_pos = headers_list == 'CONT2L'
        calmir_pos = headers_list == 'CALMIR'
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

    @property
    def catalog_headers(self):
        return [
            'FNUM', 'A_UTC', 'DATE_OBS',
            'ITIMEREQ', 'FILTER', 'GRATING', 'SLIT',
            'CALMIR', 'OBJECT',
            'ARGONL', 'NEONL', 'CONT1L', 'CONT2L',
            'A_TRGTRA', 'A_TRGTDE'
            ]

    @property
    def grouping_keys(self):
        return [
            'GRATING', 'SLIT', 'A_TRGTRA',
            'A_TRGTDE'
        ]

    @property
    def flat_grouping_keys(self):
        return self.grouping_keys[:-2]

    @property
    def flat_kw(self):
        return ['CONT1', 'CONT2']


'''
TIRSPEC instrument
'''


class TIRSPEC(Instrument):
    name = 'TIRSPEC'

    def __init__(self):
        pass

    def sort_filename_key(self, fname: str) -> int:
        basename = Path(fname).name
        number = re.search(r'(\d+)\.Z\.fits', basename)
        return int(number.group(1)) if number else -1

    def standardise_header(self, fname: str) -> dict:
        header = read_fits_header(fname)
        fnum = self.sort_filename_key(fname)
        header['FNUM'] = fnum
        return header

    def frame_select(self, fname: str) -> bool:
        return True

    def catalog_flag(self, flog_list: list) -> list:
        flog_list.append("OBJECT")
        return flog_list

    @property
    def catalog_headers(self):
        return [
            'FNUM', 'TIME', 'DATE', 'UPPER',
            'LOWER', 'SLIT', 'CALMIR', 'TARGET',
            'TCSRA', 'TCSDEC'
        ]

    @property
    def grouping_keys(self):
        return [
            'UPPER', 'LOWER', 'SLIT',
            'TCSRA', 'TCSDEC'
        ]

    @property
    def flat_kw(self):
        return ['CONT1L', 'CONT2L']


'''
Function dictionaries
'''

instrument_class = {'SpecTANSPEC': SpecTANSPEC()
                    }

# End
