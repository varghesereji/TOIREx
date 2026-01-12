#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from collections import defaultdict

from ariastrotools import divide_smoothgradient
from ariastrotools import operate_process
from ariastrotools import remove_cosmic_rays
from ariastrotools import combine_process

from .utils import extract_number_from_fname
from .utils import combine_frames
from .utils import read_txt_file
from .setups import get_logger
from .instrument import instruments


def read_files_group(fname):
    """
    Read grouped dither frame names from a text file.

    This function reads a text file containing frame names, where groups
    of frames are separated by blank lines. Each group of frame names is
    assigned to an integer key, starting from 0, and returned as a
    dictionary mapping group indices to lists of frame names.

    Parameters
    ----------
    fname : str or Path
        Path to the text file containing grouped frame names.

    Returns
    -------
    dict
        A defaultdict of lists, where each key is an integer group index
        (starting from 0) and each value is a list of frame names
        (strings) belonging to that group.

    Notes
    -----
    - Blank lines in the file are used to separate groups.
    - Leading and trailing whitespace on each line is stripped.
    - Empty lines are not included in any group.
    """
    dither_groups = defaultdict(list)
    dither_index = 0
    with open(fname, 'r') as groupfile:
        for line in groupfile:
            if line == "\n":
                dither_index += 1
                continue
            dither_groups[dither_index].append(line.strip())
    return dither_groups


def group_files_for_flatsandcals(dither_list,
                                 objflat_list,
                                 objcal_list=None,
                                 objsky_list=None):
    """
    Group dither frames by the flat and calibration frames used.

    This function associates each dither frame in ``dither_list`` with the
    corresponding flat-field frames (from ``objflat_list``) and, optionally,
    calibration frames (from ``objcal_list``). Frames are grouped by their
    associated correction frames, and the result is returned as a dictionary.

    Parameters
    ----------
    dither_list : list or array-like
        A list of science or dither frame names to be grouped.
    objflat_list : array-like of shape (N, 2)
        A 2D array where the first column contains dither frame names
        and the second column contains corresponding flat-frame names.
    objcal_list : array-like of shape (N, M), optional
        A 2D array where the first column contains dither frame names
        and the remaining columns contain corresponding calibration
        frame names. If ``None``, no calibration frames are used.
    objsky_list : array-like of shape (N, M), optional
        A 2D array where the first column contains dither frame names
        and the remaining columns contain corresponding sky frame
        frame names. If ``None ``, no sky frames are used.

    Returns
    -------
    dict
        A dictionary (defaultdict of list) mapping a string key,
        consisting of the flat-frame(s) and calibration-frame(s)
        joined by spaces, to a list of dither frames that use those
        correction frames.

    Notes
    -----
    - If multiple calibration frames exist for a dither frame, they are
      concatenated to the flat-frame list before forming the key.
    - Each group key is a string built by joining the associated flat
      and calibration frame names with spaces.
    - Dither frames without any associated flat or calibration frames
      will be grouped under the first matching flat-frame entry.
    """

    dither_list = np.array(dither_list)
    objflat_list = np.array(objflat_list)
    if objcal_list is not None:
        objcal_list = np.array(objcal_list)
    if objsky_list is not None:
        objsky_list = np.array(objsky_list)
    flat_obj = objflat_list[:, 0]
    flat_frame = objflat_list[:, 1]
    flatcorr_group = defaultdict(list)

    for oneframe in dither_list:
        # print("Dither", oneframe)
        frame_mask = flat_obj == oneframe
        oneframe_flats = flat_frame[frame_mask]
        if objcal_list is not None:
            oneframe_cals = objcal_list[frame_mask, 1:]
            if len(oneframe_cals.shape) > 1:
                oneframe_cals = oneframe_cals[0]

            oneframe_flats = np.concatenate((oneframe_flats, oneframe_cals))
        if objsky_list is not None:
            oneframe_sky = objsky_list[frame_mask, 1:]
            if len(oneframe_sky.shape) > 1:
                oneframe_sky = oneframe_sky[0]
            oneframe_flats = np.concatenate((oneframe_flats, oneframe_sky))
        if len(oneframe_flats) > 0:
            dithergroup_key = " ".join(oneframe_flats)
        else:
            dithergroup_key = oneframe_flats[0]
        flatcorr_group[dithergroup_key].append(oneframe)
    return flatcorr_group


def common_prefix(strings):
    """
    Find the longest common prefix among a list of strings.
    """
    if not strings:
        return ""

    # zip stops at the shortest string length
    prefix_chars = []
    for chars in zip(*strings):
        if all(char == chars[0] for char in chars):
            prefix_chars.append(chars[0])
        else:
            break

    return "".join(prefix_chars)


def join_frames_create_masterflat(dithergroup_dict, op_path, write_txtfname,
                                  config):
    logger = get_logger("flat_corr")
    dictkw = config['inits']['DICTKW']
    print("Write to", write_txtfname)
    writetotxt = open(write_txtfname, 'w')
    fluxexts = list(config['inputs']['FLUXEXT'])
    varexts = list(config['inputs']['VAREXT'])
    logger.info("Flux extensions: {}".format(fluxexts))
    logger.info("Variance extensions: {}".format(varexts))
    for flats_cals, objects in dithergroup_dict.items():
        common_fname_prifix = common_prefix(objects)
        comb_prifix = "{}_".format(common_fname_prifix)
        comb_filename = combine_frames(
            objects, op_path,
            instruments[dictkw]['sort_filename_key'],
            method=config['inputs']['FRAMECOMBINE'].lower(),
            op_prefix=comb_prifix,
            fluxext=fluxexts,
            varext=varexts,
            mask=instruments[dictkw]['badpixelmask'])
        # Creating masterflat
        flats_cals_list = flats_cals.strip().split(" ")
        comb_flat_fname = flats_cals_list[0]
        comb_flat_path = Path(op_path) / comb_flat_fname
        smooth_flat = create_smoothmasterflat(
            comb_flat_path,
            instruments[dictkw]['masterflat'],
            fluxext=fluxexts, varext=varexts
        )
        flats_cals_list[0] = smooth_flat
        updated_flats_cals = " ".join(flats_cals_list)
        writetotxt.write(comb_filename + " " + updated_flats_cals)
        # print("Saved", comb_filename)
    writetotxt.close()


def create_smoothmasterflat(flatfile, masterflat_fn=None,
                            fluxext=[0], varext=[1]):
    opfname = "Smooth_" + flatfile.name
    opfname_path = Path(flatfile.parent) / opfname

    if "".join(varext) == "None":
        varext = None

    if not opfname_path.exists():
        divide_smoothgradient(flatfile, opfname_path,
                              fluxext=fluxext, varext=varext)
        if masterflat_fn is not None:
            masterflat_fn(opfname_path)
    else:
        # print(opfname_path, "already exists")
        pass
    return opfname


def frame_operation(dithergroup_txtfname,
                    config,
                    op_path,
                    required="Flat"
                    ):
    """
    Perform flat-fielding or sky subtraction on groups of FITS files.

    This function reads dithergroups from a text file and processes each
    group according to the `required` parameter. For "Flat", it performs
    flat-field division, optionally removes cosmic rays, and appends
    filenames of the processed frames. For "Sky", it performs sky
    subtraction and appends the resulting filenames. All processed
    filenames are written to `write_txtfname`.

    Parameters
    ----------
    dithergroup_txtfname : str or Path
        Path to the text file containing lists of FITS filenames grouped
        by dithers.
    config : dict
        Configuration dictionary containing processing parameters. Expected
        keys:
        - 'inputs': {
            'FLUXEXT': list of flux extension indices,
            'VAREXT': list of variance extension indices,
            'REMOVECR': 'Y' or 'N'
          }
    op_path : Path
        Directory path where input FITS files are stored and output files
        should be written.
    write_txtfname : str or Path
        Path to the output text file where processed filenames will be written.
    required : str, optional
        Type of operation to perform. Supported values:
        - "Flat": flat-field division (default)
        - "Sky": sky subtraction

    Returns
    -------
    None
        Writes processed filenames to `write_txtfname` and creates
        processed FITS files in `op_path`.

    Raises
    ------
    FileNotFoundError
        If any input file listed in `dithergroup_txtfname` does not exist.
    KeyError
        If required keys in `config` are missing.
    Exception
        If processing fails in `operate_process` or `remove_cosmic_rays`.
    """
    dithergroups = read_txt_file(dithergroup_txtfname)
    # print(dithergroups)
    # Going through each dither group
    fluxexts = list(config['inputs']['FLUXEXT'])
    varexts = list(config['inputs']['VAREXT'])
    if "".join(varexts) == "None":
        varexts = None
    new_list = []
    txtfile_line = []
    for dgroup in dithergroups:
        sci_fname = op_path / dgroup[0]

        if required == "Flat":
            op_fname = sci_fname.stem + "_FC.fits"
            operation = "/"
            secondframe_fname = op_path / dgroup[1]
        elif required == "Sky":
            op_fname = sci_fname.stem + "_Skysubtr.fits"
            operation = "-"
            secondframe_fname = op_path / dgroup[-1]
        op_fname = op_path / op_fname
        # print(sci_fname, flat_fname, op_fname)
        operate_process(sci_fname, secondframe_fname,
                        op_fname, operation=operation,
                        fluxext=fluxexts,
                        varext=varexts)
        new_list.append(op_fname.name)
        if required == "Flat":
            if (config['inputs']['REMOVECR'] == 'Y'):
                crop_fname = op_fname.stem+"_CR.fits"
                crop_fname = op_path / crop_fname
                remove_cosmic_rays(op_fname,
                                   crop_fname,
                                   fluxext=fluxexts,
                                   varext=varexts)
                new_list[0] = crop_fname.name
            if len(dgroup[2:]) != 0:
                # For photometry, there will be lamp.
                # That is writinig to the text editor.
                new_list = new_list + dgroup[2:]
                new_list_str = " ".join(new_list)
            else:
                new_list_str = new_list[0]
        elif required == "Sky":
            # In the list dgroup, if sky subtraction is
            # activated in config file, last entry will be
            # sky. That is removed here.
            new_list = new_list + dgroup[1:-1]
            new_list_str = " ".join(new_list)
        txtfile_line.append(new_list_str+"\n")
    # writetotxt.close()
    return txtfile_line


def mediancomb_sky_subtr(frames_list, opdir, config, group):
    frames_list = [opdir / i for i in frames_list]
    print(frames_list)
    combkg_fname = opdir / "mediancomb_bkg{}.fits".format(group)
    combine_process(frames_list,
                    combkg_fname,
                    method='median',
                    fluxext=list(config['inputs']['FLUXEXT']),
                    varext=list(config['inputs']['VAREXT'])
                    )
    for dframe in frames_list:
        operate_process(dframe, combkg_fname,
                        dframe, operation='-',
                        fluxext=list(config['inputs']['FLUXEXT']),
                        varext=list(config['inputs']['VAREXT'])
                        )


def frame_correction(config, dirname):
    """
    Perform flat-field and calibration corrections for grouped dither frames.
    Sky subtraction will be done if specified in config file.

    This function locates the text files containing grouped dither frames,
    reads the corresponding flat-field and calibration frame lists, and
    applies grouping logic to associate each dither frame with its correction
    frames. Results are logged and printed, but not written to disk.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - ``outputs.OP_DIR`` : str
            Path to the output directory containing group text files.
        - ``inits.TODO`` : str
            Flag indicating whether to process calibration frames.
            If set to ``"S"``, calibration files are read; otherwise,
            no calibration is applied.
    dirname : str
        Name of the subdirectory under ``OP_DIR`` that contains the group
        text files.

    Returns
    -------
    None
        The function does not return any value. It prints diagnostic
        information and logs processing details using the configured logger.

    Notes
    -----
    - Input group files must follow the naming convention
      ``ObjectsToCombine_group*.txt``.
    - Flat and calibration frame lists are expected to follow the naming
      convention:
        * ``Objects_finalflats_groupN.txt``
        * ``Objects_finalcals_groupN.txt`` (if applicable)
    - For each dither group, the function calls
      :func:`group_files_for_flatsandcals` to form associations between
      dither frames and their flat/calibration frames.
    - The logger name used is ``"flat_corr"``.
    """
    logger = get_logger("flat_corr")
    txtfile_re = "ObjectsToCombine_group*.txt"
    op_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(op_path.glob(txtfile_re))
    for f in files_list:
        number = extract_number_from_fname(f.name)
        finalflat_txtfile = "Objects_finalflats_group{}.txt".format(number[0])
        finalflat_list = read_txt_file(op_path / finalflat_txtfile)
        if config['inits']['TODO'] == "S":
            finalcal_txtfile = "Objects_finalcals_group{}.txt".format(
                number[0])
            finalcal_list = read_txt_file(op_path / finalcal_txtfile)
        else:
            finalcal_list = None
        if config['inputs']['SKY'] == 'Y':
            finalsky_txtfile = "Objects_finalsky_group{}.txt".format(
                number[0])
            finalsky_list = read_txt_file(op_path / finalsky_txtfile)
        else:
            finalsky_list = None
        print("Group numbr running:", int(number[0]))
        logger.info("Calling file"+f.name)
        if config['inits']['TODO'] == 'P':
            # For photmetry extraction, there is no other calibration after
            # flat correction. Therefore, no need to keep each dither frame
            # in separate txt files. In spectroscopy. We are keeping different
            # text file for each dither position is because the lamp file
            # may be different for different frame, even for same target.
            # So, in same dither position, the lines in text editor
            # can be used for different lamp/flat combinations.
            # In photometry, there is no wavelength calibration.
            # Therefore, we can combine the frames in same dither position
            # after flat correction, and keep same text file for all
            # dither frames.

            clean_frame_txtfname = "Clean_frame_group{}_d{}.txt".format(
                    number[0], "Full")
            clean_frame_txtfname = op_path / clean_frame_txtfname
            writetotxt_cleanframe = open(clean_frame_txtfname, 'w')
            # If observed sky is not available,
            # median-combine all dither frames and subtract that from
            # combined-flat corrected frame.
            allditherpos_list = []
        dither_groups = read_files_group(f)
        for n, samepos in dither_groups.items():
            logger.info("Dither pos {}: {}".format(n, samepos))
            print("Dither group", n)
            flatcorr_group = group_files_for_flatsandcals(samepos,
                                                          finalflat_list,
                                                          finalcal_list,
                                                          finalsky_list)
            combobj_flat_txtfname = "Combobj_flat_group{}_d{}.txt".format(
                number[0], n)
            combobj_flat_txtfname = op_path / combobj_flat_txtfname
            join_frames_create_masterflat(flatcorr_group, op_path,
                                          combobj_flat_txtfname,
                                          config)
            if config['inputs']['SKY'] == 'Y':
                # Sky subtraction
                skysubtr_txtfname = "Skysubtr_frame_group{}_d{}.txt".format(
                    number[0], n)
                skysubtr_txtfname = op_path / skysubtr_txtfname
                skysubtr_writeto = open(skysubtr_txtfname)
                txtfile_line_sky = frame_operation(combobj_flat_txtfname,
                                                   config,
                                                   op_path,
                                                   required="Sky")
                full_lines_sky = "".join(txtfile_line_sky)
                writetotxt_cleanframe.write(full_lines_sky)
                skysubtr_writeto.close()
                combobj_flat_txtfname = skysubtr_txtfname
            if config['inits']['TODO'] == 'S':
                clean_frame_txtfname = "Clean_frame_group{}_d{}.txt".format(
                    number[0], n)
                clean_frame_txtfname = op_path / clean_frame_txtfname
                writetotxt_cleanframe = open(clean_frame_txtfname, 'w')
            # Flat correction
            txtfile_line = frame_operation(combobj_flat_txtfname,
                                           config, op_path
                                           )

            if config['inits']['TODO'] == 'P':
                # Combine all flat-corrected frames in
                # same dither positions.
                if len(txtfile_line) > 1:
                    fname_stems = [Path(i).stem for i in txtfile_line]
                    comb_dither_fname = "+".join(fname_stems) + ".fits"
                     # adding path to each frame name
                    fnames_path = [op_path / i for i in txtfile_line]
                    combine_process(fnames_path,
                                    op_path / comb_dither_fname,
                                    method=config['inputs']['FRAMECOMBINE'],
                                    fluxext=list(config['inputs']['FLUXEXT']),
                                    varext=list(config['inputs']['VAREXT'])
                                    )
                    full_lines = comb_dither_fname
                else:
                    full_lines = txtfile_line[0]
                allditherpos_list.append(full_lines.strip())
                full_lines = 'd{} : '.format(n) + full_lines
            else:
                full_lines = "".join(txtfile_line)
            writetotxt_cleanframe.write(full_lines)

            if config['inits']['TODO'] == 'S':
                writetotxt_cleanframe.close()
        if config['inits']['TODO'] == 'P':
            # For photometry, the text editor was open before the for loop.
            # That is closed here.
            mediancomb_sky_subtr(allditherpos_list, op_path, config, number[0])
            writetotxt_cleanframe.close()

# End
