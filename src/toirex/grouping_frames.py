#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
from pathlib import Path
import re

from .setups import get_logger

from .obscatalog import read_catalog
from .instrument import instruments
from .utils import open_in_editor
from .utils import extract_fname_prefix


def remove_repeated_values(candidate_list):
    """
    Remove duplicate elements from a list, with special handling for
    NumPy arrays.

    This function iterates through the input list and removes repeated
    elements. Unlike the built-in `set`, it preserves the original order
    and works with elements that are NumPy arrays by comparing their
    contents (via `tolist()`) rather than object identity.

    Parameters
    ----------
    candidate_list : list
        Input list potentially containing repeated elements. Elements can be
        of arbitrary type, including NumPy arrays.

    Returns
    -------
    filtered_list : list
        A new list with repeated elements removed, preserving the first
        occurrence of each unique element.

     Notes
    -----
    - For NumPy arrays, equality is determined by comparing the result of
      `.tolist()`, so arrays with the same contents but different memory
      locations are considered duplicates.
    - The function preserves the order of elements, unlike `set()`.

    Examples
    --------
    >>> import numpy as np
    >>> arr1 = np.array([1, 2, 3])
    >>> arr2 = np.array([1, 2, 3])
    >>> candidate_list = [arr1, arr2, [4, 5], [4, 5], "a", "a"]
    >>> remove_repeated_values(candidate_list)
    [array([1, 2, 3]), [4, 5], 'a']
    """

    filtered_list = []
    for cand in candidate_list:
        cand_list = cand.tolist() if isinstance(cand, np.ndarray) else cand
        if not any(
                (x.tolist() if isinstance(x, np.ndarray) else x) == cand_list
                for x in filtered_list
        ):
            filtered_list.append(cand)
        else:
            pass
    return filtered_list


def select_flats(catalog_dict, flats_keys, flat_flag):
    """
    Select and group flat frames from a catalogue.

    This function extracts flat frames from the input catalogue based on
    classification flags and groups them according to a set of catalogue
    keys (e.g., grism, filter, order). Duplicate groups are removed using
    `remove_repeated_values`. The grouped flats are returned as a dictionary
    mapping a concatenated string of key values to the corresponding flat
    file names.

    Parameters
    ----------
    catalog_dict : dict
        Dictionary containing the catalogue. Must include:
        - ``'FNAME'`` : list or array of file names
        - ``'FLAG'`` : list or array of classification flags
        - Additional keys specified in `flats_keys`.

    flats_keys : list of str
        List of catalogue dictionary keys used to define unique groups
        (e.g., ["GRISM", "FILTER"]).

    flat_flag : list of str
        List of flags that identify flat frames in the catalogue.

    Returns
    -------
    grouped_flats : dict
        Dictionary where:
        - keys are strings obtained by joining the values of the group-defining
          keys with a space (e.g., "GRISM1 FILTER2"),
        - values are arrays of file names corresponding to that group of flats.

    Notes
    -----
    - The grouping is done by extracting values from `catalog_dict` at
    positions
      where `FLAG` matches `flat_flag`.
    - Duplicate groups are removed using `remove_repeated_values`.
    - The function assumes that the catalogue dictionary values can be indexed
      and broadcast into NumPy arrays.

    Examples
    --------
    >>> catalog_dict = {
    ...     "FNAME": ["flat1.fits", "flat2.fits", "sci1.fits"],
    ...     "FLAG":  ["FLAT", "FLAT", "SCIENCE"],
    ...     "FILTER": ["R", "R", "R"],
    ...     "GRISM":  ["G1", "G1", "G1"]
    ... }
    >>> flats_keys = ["FILTER", "GRISM"]
    >>> flat_flag = ["FLAT"]
    >>> select_flats(catalog_dict, flats_keys, flat_flag)
    {'R G1': array(['flat1.fits', 'flat2.fits'], dtype='<U10')}
    """

    group_entries = []
    fnames = np.array(catalog_dict['FNAME'])
    flags = catalog_dict['FLAG']
    flat_mask = np.array([True if i in flat_flag else False for i in flags])
    for keys in flats_keys:
        # print(keys)
        catalog_keys = np.array(catalog_dict[keys])
        group_entries.append(catalog_keys[flat_mask])
    group_entries = np.array(group_entries).T
    reduced_groups = remove_repeated_values(group_entries)
    flat_fnames = fnames[flat_mask]
    grouped_flats = {}
    for n, r in enumerate(reduced_groups):
        # print(r)
        mask = group_entries == r
        matchings = np.sum(mask, axis=1) == len(r)
        matched_fnames = flat_fnames[matchings]
        # print(matched_fnames)
        dictkw = " ".join(r)
        grouped_flats[dictkw] = matched_fnames
    # print(grouped_flats)
    return grouped_flats


def ordered_keys(catalog_dict, config, grouping_keys, flats_keys, flat_flag):
    """
    Group files by specified catalogue keys and append matching flats.

    This function groups catalogue entries according to a set of header keys
    (e.g., grism, filter, order). For each unique group, it collects the
    corresponding file names and appends any flat frames that match based on
    `flats_keys`. The result is a dictionary mapping group indices to arrays
    of file names (science + flats).

    Parameters
    ----------
    catalog_dict : dict
        Dictionary containing the catalogue. Must include:
        - ``'FNAME'`` : list or array of file names.
        - Additional keys specified in `grouping_keys` and `flats_keys`.

    config : dict
        Configuration dictionary (currently unused in this function, but passed
        for consistency with other pipeline functions).

    grouping_keys : list of str
        Catalogue dictionary keys used to define the main grouping of science
        files (e.g., ["OBJECT", "FILTER", "GRISM"]).

       flats_keys : list of str
        Catalogue dictionary keys used to group flat frames.

    flat_flag : list of str
        List of flags identifying flat frames in the catalogue.

    Returns
    -------
    grouped_files : dict
        Dictionary where:
        - keys are integer group indices (0, 1, 2, …),
        - values are arrays of file names corresponding to that group,
          including any matching flat frames.

    Notes
    -----
    - Groups are formed by unique combinations of values in `grouping_keys`.
    - Flat frames are grouped separately using `select_flats` and then merged
      into the corresponding science groups if their key values match.
    - Duplicate groups are removed using `remove_repeated_values`.
    - The function preserves the order of first occurrences.

    Examples
    --------
    >>> catalog_dict = {
    ...     "FNAME": ["sci1.fits", "sci2.fits", "flat1.fits"],
    ...     "FLAG":  ["SCIENCE", "SCIENCE", "FLAT"],
    ...     "OBJECT": ["StarA", "StarA", "Lamp"],
    ...     "FILTER": ["R", "R", "R"],
    ...     "GRISM":  ["G1", "G1", "G1"]
    ... }
    >>> grouping_keys = ["OBJECT", "FILTER", "GRISM"]
    >>> flats_keys = ["FILTER", "GRISM"]
    >>> flat_flag = ["FLAT"]
    >>> grouped = ordered_keys(catalog_dict, {}, grouping_keys, flats_keys,
        flat_flag)
    >>> list(grouped.keys())
    [0]   # one science group
    >>> grouped[0]
    array(['sci1.fits', 'sci2.fits', 'flat1.fits'], dtype='<U10')
    """

    # Recognizing the position of flats_keys in grouping_keys.
    # This will filter out the entries for flat.
    # If this is not there, repeated entries will be grouped.
    flat_key_pos = np.array([True if i in flats_keys else False
                             for i in grouping_keys])
    fnames = np.array(catalog_dict['FNAME'])

    group_entries = []
    #  select_flats(catalog_dict, flats_keys, flat_flag)
    for keys in grouping_keys:
        group_entries.append(catalog_dict[keys])
    group_entries = np.array(group_entries).T
    reduced_groups = remove_repeated_values(group_entries)
    grouped_flats = select_flats(catalog_dict, flats_keys, flat_flag)
    grouped_files = defaultdict(list)

    for n, r in enumerate(reduced_groups):
        mask = group_entries == r
        matchings = np.sum(mask, axis=1) == len(r)
        matched_fnames = fnames[matchings]
        for flat_kw, flat_fnames in grouped_flats.items():
            flat_kwlist = flat_kw.strip().split(" ")
            filtered_r = np.array(r)[flat_key_pos]
            entry_mask = np.array([
                True if i in flat_kwlist else False for i in filtered_r
            ])
            if np.sum(entry_mask) == len(flat_kwlist):
                matched_fnames = np.concatenate((
                    matched_fnames, np.array(flat_fnames
                                             )))
        grouped_files[n] = matched_fnames
    return grouped_files


def grouping_items(config, dirname, catalogue_dict=None,
                   open_editor=True):
    """
    Group catalogue files by observing setup while excluding flat frames.

    This function reads the catalogue for the specified directory, groups
    frames according to the instrument-specific grouping keys, and associates
    each frame with its classification flag. A human-readable summary of the
    groups is written to ``Grouped_txtfile.txt``, and the grouped catalogue is
    returned as a nested dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

        The following keys are required:

        - ``config['inits']['DICTKW']``: Instrument identifier used to select
          the appropriate grouping rules.
        - ``config['outputs']['OP_DIR']``: Output directory where the grouping
          summary is written.

    dirname : str or pathlib.Path
        Subdirectory of the output directory corresponding to the dataset.

    Returns
    -------
    dict
        Nested dictionary containing the grouped catalogue.

        The outer dictionary maps group numbers to dictionaries of frame types.
        Each inner dictionary maps classification flags (e.g. ``OBJECT``,
        ``ARGON``, ``FLAT``) to lists of filenames.

    Notes
    -----
    Flat frames are excluded using the instrument-specific ``flat_kw`` keyword.

    Grouping relies on the following components:

    - ``read_catalog()`` to load the catalogue.
    - ``instruments[dictkw]['grouping_keys']`` to define observing groups.
    - ``instruments[dictkw]['flat_grouping_keys']`` to associate flat frames.
    - ``ordered_keys()`` to construct the grouped catalogue.

    The function also writes ``Grouped_txtfile.txt`` to
    ``<config['outputs']['OP_DIR']>/<dirname>``.

    A placeholder exists for automatically associating continuum flats in a
    future version.

    Examples
    --------
    >>> config = {
    ...     "inits": {"DICTKW": "SpecTANSPEC"},
    ...     "outputs": {"OP_DIR": "Reduced_data"},
    ... }
    >>> groups = grouping_items(config, "20240908")
    >>> sorted(groups.keys())
    [0, 1, 2]
    """
    if catalogue_dict is None:
        catalogue_dict = read_catalog(dirname, config)
    catalog_fnames = np.array(catalogue_dict['FNAME'])
    flags = catalogue_dict['FLAG']
    dictkw = config['inits']['DICTKW']

    flat_flag = instruments[dictkw]['flat_kw']

    groups_dict = {}
    opdir = Path(config['outputs']['OP_DIR']) / \
        dirname
    txt_fname = opdir / "Grouped_txtfile.txt"
    grouped_txt_file = open(txt_fname, 'w')
    grouping_keys = instruments[dictkw]['grouping_keys']
    flat_keys = instruments[dictkw]['flat_grouping_keys']
    ordered_dict = ordered_keys(catalogue_dict, config,
                                grouping_keys, flat_keys, flat_flag)
    opfilename_prefix = open(opdir / "Filename_suggestions.txt", 'w')
    for order, fnames in ordered_dict.items():
        subgroups_dict = defaultdict(list)
        group_header = "*" * 10 + "Group {} ".format(order) + "*" * 10 + "\n"
        grouped_txt_file.write(group_header)
        for crorder, fname in enumerate(catalog_fnames):
            if fname in fnames:
                # print(crorder, fname, 'in group', order)
                subgroups_dict[flags[crorder]].append(fname)
        if 'OBJECT' not in list(subgroups_dict.keys()):
            grouped_txt_file.write("\n No object in this group \n")
            continue

        opfilename_prefix.write(
            "{} : {}\n".format(
                order,
                extract_fname_prefix(
                    subgroups_dict['OBJECT'][0],
                    instruments[dictkw]['fname_regexp']
                )
            )
        )
        for keys, fnames in subgroups_dict.items():
            grouped_txt_file.write(
                "{}: \n{}\n".format(keys, "\n".join(fnames))
            )
            grouped_txt_file.write("\n")
        grouped_txt_file.write("\n")
        groups_dict[order] = subgroups_dict
    # A function to add continuum flats here.
    grouped_txt_file.close()
    opfilename_prefix.close()
    if (
            config['inits']['MODE'] == "AUTO"
            and config['inits']['TIMESERIES'] == 'N'
    ):
        open_editor = False
    if open_editor:
        open_in_editor(txt_fname, config)
    return groups_dict


def grouping_with_re(config, dirname):
    """
    Group catalogue entries using user-specified regular expressions.

    This function reads a catalogue of image frames from the given directory
    and filters them based on user-defined regular expressions. Users are first
    prompted to input regex patterns for science frames and flat frames, and
    optionally for lamp frames (depending on the configuration). The matched
    frames are collected into a reduced catalogue and then grouped by
    instrument-specific keywords (e.g., SLIT, GRATING, FILTER) via
    `grouping_items`.

    Parameters
    ----------
    config : dict
        Configuration dictionary with at least the following keys:

        - ``config['inits']['TODO']`` : str
            Indicates the processing mode. If set to ``'S'``, the user will be
            prompted to enter regex rules for lamp frames as well.

        - ``config['inits']['DICTKW']`` : str
            Key used to identify the instrument class from `instrument_class`.
            This is needed to obtain instrument-specific keywords (e.g.,
            lamp keywords).

    dirname : str
        Path to the directory containing the catalogue and associated files.

    Returns
    -------
    grouped_dict : dict
        A dictionary of grouped catalogue entries, as produced by
        `grouping_items`. The grouping is based on instrument-specific
        parameters (SLIT, GRATING, FILTER, etc.) after filtering by the
        user-provided regex rules.

    Notes
    -----
    - The function is interactive: it prompts the user to enter regular
      expressions for selecting science, flat, and (if applicable) lamp frames.
    - Frames that match at least one of the provided regex rules are collected.
    - Grouping is performed by calling the external `grouping_items` function.
    - Regex documentation reference:
      http://docs.python.org/2/howto/regex.html#regex-howto

    Examples
    --------
    >>> config = {
    ...     'inits': {
    ...         'TODO': 'S',
    ...         'DICTKW': 'TIRSPEC'
    ...     }
    ... }
    >>> dirname = "/path/to/data"
    >>> grouped = grouping_with_re(config, dirname)
    Enter the regular expression for SCIECNE frames: .*M31.*
    Enter the regular expression for FLAT frames: .*continuum.*
    Enter the regular expression for LAMP frames: .*argon.*
    >>> print(grouped.keys())
    dict_keys(['FNAME', 'SLIT', 'FILTER', 'GRATING', ...])
    """

    catalogue_dict = read_catalog(dirname, config)
    fnames_full = catalogue_dict['FNAME']
    fnums_full = catalogue_dict['FNUM']
    print('*'*10)
    print('For Regular Expression rules See:', end=" ")
    print('http://docs.python.org/2/howto/regex.html#regex-howto')
    print('Some examples of typical input are shown below')
    print('.*M31.* is the regular expression to select', end=" ")
    print('all the objects lines which has "M31" in it.')
    print('NB: Even you enter the regular expression, the objects', end=" ")
    print('will be grouped based on SLIT, GRATING, FILTER etc')
    enter_object = input("Enter the regular expression for SCIECNE frames:")
    object_re, object_fnums = reading_re(enter_object)
    re_dict = {object_re: object_fnums}
    enter_flats = input("Enter the regular expression for FLAT frames:")
    flats_re, flat_fnums = reading_re(enter_flats)

    re_dict[flats_re] = flat_fnums

    if config['inits']['TODO'] == 'S':
        dictkw = config['inits']['DICTKW']
        lamp_keys = instruments[dictkw]['lamp_kw']
        for lamp in lamp_keys:
            enter_lamp = input(
                "Enter the regular expression for {} frames:".format(lamp)
            )
            lamp_re, lamp_fnums = reading_re(enter_lamp)
            re_dict[lamp_re] = lamp_fnums
    selected_objects_dict = defaultdict(list)
    catalog_kws = catalogue_dict.keys()
    for res_ent, fnums in re_dict.items():
        fnum_mask = (fnums_full >= fnums[0]) & \
            (fnums_full <= fnums[1])
        fnames = fnames_full[fnum_mask]
        for n, imgline in enumerate(fnames):
            # Converting the entered re string to re object
            res = re.compile(r''+res_ent)
            if res.search(imgline) is not None:
                for dictkws in catalog_kws:
                    selected_objects_dict[dictkws].append(
                        catalogue_dict[dictkws][fnum_mask][n])
    grouped_dict = grouping_items(config, dirname,
                                  catalogue_dict=selected_objects_dict,
                                  open_editor=False)
    return grouped_dict


def reading_re(entered_re):
    """
    Function to read the entered
    regular expression and separate FNUM from the
    entered item.
    """
    if len(entered_re.strip().split(" ")) == 3:
        re_list = entered_re.strip().split(" ")
        object_re = re_list[0]
        start_fnum = int(re_list[1])
        end_fnum = int(re_list[-1])
    else:
        object_re = entered_re
        start_fnum = 0
        end_fnum = 10000000
    fnum_list = [start_fnum, end_fnum]
    return object_re, fnum_list

# End
