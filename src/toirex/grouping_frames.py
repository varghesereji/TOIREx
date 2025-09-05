#!/usr/bin/env python3
from collections import defaultdict
import numpy as np

from .instrument import functions_dict


def remove_repeated_values(candidate_list):
    '''
    Input
    --------
    candidate_list: List to remove the repeated elements
    Return
    --------
    filtered_list: List which repeated elements removed from candidate_list
    '''

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


def ordered_keys(catalog_dict, config):
    '''
    This function is to group the files based on header.
    '''
    # print(catalog_dict)
    # catalog_list = np.array(catalog_list)
    dictkw = config['inits']['DICTKW']  # Calling the directory keyword
    grouping_keys = functions_dict[dictkw]['grouping_keys']
    dict_keys = list(catalog_dict.keys())

    fnames = np.array(catalog_dict['FNAME'])
    group_entries = []
    for keys in grouping_keys:
        group_entries.append(catalog_dict[keys])
    group_entries = np.array(group_entries).T
    reduced_groups = remove_repeated_values(group_entries)
    grouped_files = defaultdict(list)
    for n, r in enumerate(reduced_groups):
        # print(n, r)
        mask = group_entries == r
        # print(mask, np.sum(mask, axis=1))
        matchings = np.sum(mask, axis=1) == len(r)
        # matched_entries = group_entries[matchings]
        matched_fnames = fnames[matchings]
        grouped_files[n] = matched_fnames
    return grouped_files


def grouping_items(ordered_dict, catalogue_dict):
    '''
    ordered_dict: dictionary with grouped filenames.
    dict keys will be the group number.
    catalogue_dict: ditonary of full catalogue.
    '''

    catalog_fnames = catalogue_dict['FNAME']
    flags = catalogue_dict['FLAG']
    groups_dict = {}
    for order, fnames in ordered_dict.items():
        subgroups_dict = defaultdict(list)
        for crorder, fname in enumerate(catalog_fnames):
            if fname in fnames:
                # print(crorder, fname, 'in group', order)
                subgroups_dict[flags[crorder]].append(fname)
        groups_dict[order] = subgroups_dict
    # A function to add continuum flats here.
    return groups_dict


# End
