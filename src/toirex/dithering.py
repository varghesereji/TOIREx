#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for identifying, aligning, subtracting, and combining dithered
astronomical observations.

This module provides routines for processing dithered imaging and
spectroscopic observations within the TOIREx pipeline. It supports both
spectroscopic dither subtraction and imaging dither combination, including
automatic or manual frame alignment, image registration using phase
cross-correlation, frame shifting, and image stacking.

For spectroscopic observations, the module groups frames by dither position,
allows interactive specification of subtraction pairs, performs pairwise
subtraction of dithered images, and generates input lists for subsequent
spectral extraction.

For imaging observations, the module determines relative image shifts either
interactively from user-selected reference sources or automatically through
phase cross-correlation. Aligned frames are combined into a single science
image, after which an interactive World Coordinate System (WCS) calibration is
performed to prepare the image for source extraction and astrometric analysis.

The module also includes utility functions for reading dither information,
organizing frames by observing group, applying median filtering for image
registration, and generating metadata required by later pipeline stages.

Notes
-----
Most high-level routines in this module are interactive and may prompt the
user for input, including dither subtraction instructions, reference target
selection, WCS target identification, and editing of WCS catalogues.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import median_filter

from skimage import registration
import shutil
from ariastrotools import operate_process
from ariastrotools import combine_process
from ariastrotools import shifting_frame
from ariastrotools import masking_frame

from .setups import get_logger

from .utils import read_fits_data
from .utils import text_to_dict
from .utils import extract_number_from_fname
from .utils import read_txt_file
from .utils import get_filename
from .utils import write_asciitable
from .utils import open_in_editor
from .image_utils import wcs_correction

from .plottings import imageplot

# ----------------#
# Identify dither #
# ----------------#


def filter_image(frame, size=(10, 10)):
    """
    Apply a median filter to an image frame.

    This function applies a median filter to reduce noise in the input image.
    The filter size can be specified for both dimensions.

    Parameters
    ----------
    frame : ndarray
        Input image data (2D array).
    size : tuple of int, optional
        Size of the median filter window along each axis.
        Default is (10, 10).

    Returns
    -------
    ndarray
        Filtered image of the same shape as the input.

    Example
    -------
    >>> filtered = filter_image(image, size=(5, 5))
    """
    filtered_image = median_filter(frame, size=size)
    return filtered_image


def find_shift(frame_1, frame_2, config,
               flatframe=None,
               badpixelmask=None):
    """
    Compute the relative shift between two FITS images.

    This function determines the translational offset required to align one
    image with another using phase cross-correlation. Input images may be
    supplied either as FITS filenames or as already loaded arrays. Optional
    flat-field correction and bad-pixel masking are applied before the
    images are cropped, filtered, and registered.

    Parameters
    ----------
    frame_1 : str, pathlib.Path, or ndarray
        Reference image or path to the reference FITS file.
    frame_2 : str, pathlib.Path, or ndarray
        Moving image or path to the FITS file to be aligned with
        ``frame_1``.
    config : dict
        Pipeline configuration dictionary. The image crop region is read
        from ``config['dither']['CROP']`` and the phase cross-correlation
        upsampling factor from ``config['dither']['UPSAMPLE']``.
    flatframe : str, pathlib.Path, ndarray, callable, or None, optional
        Flat-field image used to normalize both input frames before
        registration. If a callable is supplied, it must return the flat
        frame corresponding to ``frame_1``. If a filename is supplied, the
        FITS file is read automatically. If ``None`` (default), no
        flat-field correction is applied.
    badpixelmask : str, pathlib.Path, ndarray, callable, or None, optional
        Bad-pixel mask applied to both images before registration. If a
        callable is supplied, it must return the mask. If ``None``
        (default), no bad-pixel masking is performed.

    Returns
    -------
    tuple
        Output returned by
        ``skimage.registration.phase_cross_correlation()`` consisting of:

        shift : ndarray
            Estimated translational shift ``(y_shift, x_shift)`` required
            to align ``frame_2`` with ``frame_1``.
        error : float
            Translation-invariant normalized root-mean-square error.
        diffphase : float
            Global phase difference between the two images, in radians.

    Notes
    -----
    - Flat-field correction is applied before bad-pixel masking.
    - Image registration is performed only on the cropped region specified
      by ``config['dither']['CROP']``.
    - NaN values introduced by bad-pixel masking are excluded from the
      phase cross-correlation using validity masks.

    Examples
    --------
    >>> config = {
    ...     'dither': {
    ...         'CROP': '10 100 20 200',
    ...         'UPSAMPLE': '50'
    ...     }
    ... }
    >>> shift, error, diffphase = find_shift(
    ...     "image1.fits",
    ...     "image2.fits",
    ...     config
    ... )
    """
    if flatframe is not None:
        if callable(flatframe):
            flatframe = flatframe(frame_1)
        if isinstance(flatframe, (str, Path)):
            flatframe = read_fits_data(flatframe)

    if isinstance(frame_1, (str, Path)):
        frame_1 = read_fits_data(frame_1)
    if isinstance(frame_2, (str, Path)):
        frame_2 = read_fits_data(frame_2)
    if flatframe is not None:
        frame_1 = frame_1 / flatframe
        frame_2 = frame_2 / flatframe

    if badpixelmask is not None:
        if callable(badpixelmask):
            badpixelmask = badpixelmask()

        frame_1 = masking_frame(frame_1, badpixelmask, method='nan')
        frame_2 = masking_frame(frame_2, badpixelmask, method='nan')

    crop = config['dither']['CROP']
    upsample_factor = int(config['dither']['UPSAMPLE'])
    crop = list(map(int, crop.split()))
    crop_yb, crop_yt, crop_xl, crop_xr = crop
    filt_img1 = filter_image(frame_1[crop_yb:crop_yt, crop_xl:crop_xr])
    filt_img2 = filter_image(frame_2[crop_yb:crop_yt, crop_xl:crop_xr])
    img1_mask = ~np.isnan(filt_img1)
    img2_mask = ~np.isnan(filt_img2)
    difference = registration.phase_cross_correlation(
        reference_image=filt_img1,
        moving_image=filt_img2,
        upsample_factor=upsample_factor,
        reference_mask=img1_mask,
        moving_mask=img2_mask
    )
    return difference

# ---------------------------- #
# Common functions for dithers #
# ---------------------------- #


def dithers_and_groups(groups_dithers_list):
    """
    Organize a list of [group, dither] pairs into a dictionary mapping groups
    to dithers.

    Takes a list of two-element lists or tuples where the first element is a
    group identifier and the second element is a dither number.
    Returns a defaultdict where each key is a group and the value is a list
    of dithers belonging to that group.

    Args:
        groups_dithers_list (list): List of [group, dither] pairs.

    Returns:
        defaultdict: Dictionary mapping group identifiers to lists of dither
             numbers.

    Example:
        >>> dithers_and_groups([[1, 0], [1, 1], [2, 0]])
        defaultdict(<class 'list'>, {1: [0, 1], 2: [0]})
    """
    groups_dithers_dict = defaultdict(list)
    for sublist in groups_dithers_list:
        groups_dithers_dict[sublist[0]].append(sublist[1])
    return groups_dithers_dict


def get_dithers(opdir, mode="S"):
    """
    Retrieve groups and their associated dithers from the output directory.

    Searches the output directory for dither text files matching the pattern
    'Clean_frame_group*_d*.txt', extracts group and dither numbers from each
    filename, and organizes them into a structure mapping groups to dithers.

    Args:
        opdir (Path): Path to the output directory containing dither text
             files.

    Returns:
        dict or suitable structure: Mapping of group identifiers to lists of
        associated dither numbers.

    Notes:
        Depends on helper functions `extract_number_from_fname` to parse
        numbers from filenames and `dithers_and_groups` to organize the
        mappings.
    """
    if mode == "P":
        dithertxt_fname = "Clean_frame_group*_dFull.txt"
    elif mode == "S":
        dithertxt_fname = "Clean_frame_group*_d*.txt"
    dither_txtfiles = list(opdir.glob(dithertxt_fname))
    groups_dithers = []
    print(dither_txtfiles)
    for group in dither_txtfiles:
        # Extract the group and dither indices from the filename.
        # For phtometry, there will be only group number.
        numbers = extract_number_from_fname(group.name)
        if mode == "P":
            groups_dithers.append(numbers[0])
        elif mode == "S":
            groups_dithers.append(numbers)
    if mode == "P":
        return groups_dithers
    elif mode == "S":
        return dithers_and_groups(groups_dithers)


# ----------------------------- #
# Subtract dithers in spectra   #
# ----------------------------- #


def read_dither_txtfile(pairstr, group,
                        opdir):
    """
    Read the dither text file for a specific group and dither position.

    Converts a dither character (A-Z) to its numeric position, constructs
    the filename, reads the corresponding text file from the output directory,
    and returns its lines.

    Args:
        pairstr (str or int): Dither identifier as a letter (e.g., 'A') or
            numeric position.
        group (str or int): Frame group identifier.
        opdir (Path): Directory path containing the dither text files.

    Returns:
        list: Lines read from the specified dither text file.

    Notes:
        Assumes existence of a helper function `read_txt_file` that reads
        the content of a text file and returns the lines.
    """
    if isinstance(pairstr, str):
        ditherpos_num = ord(pairstr) - ord("A")
    else:
        ditherpos_num = pairstr
    dithertxt = "Clean_frame_group{}_d{}.txt".format(group,
                                                     ditherpos_num)
    txtlines = read_txt_file(opdir / dithertxt)
    return txtlines


def copy_nopair_frames(dither, group, opdir,
                       opfilename, writeto):
    """
    Copy single dither frames without subtraction and log the operations.

    Reads the dither text file for the specified dither and group, copies
    each listed frame to a new file with an optional index suffix, and writes
    a descriptive line to the given writable file.

    Args:
        dither (str): Identifier of the dither to copy.
        group (str/int): Frame group identifier.
        opdir (Path): Directory path where the frames and output files are
            located.
        opfilename (str): Base filename prefix for output files.
        writeto (file-like object): Open file handle used to write operation
          logs.

    Writes:
        Lines describing each copied frame to the `writeto` file.

    Notes:
        Uses helper function `read_dither_txtfile` to obtain frame info.
        Each copied file is suffixed with an index if multiple frames exist.
    """
    txtlines_full = read_dither_txtfile(dither,
                                        group,
                                        opdir)
    print("txtlines_full", txtlines_full)
    for n, line in enumerate(txtlines_full):
        scfname = line[0]
        if len(txtlines_full) > 1:
            opfilename = opfilename + str(n)
        opfilename = opfilename + ".fits"
        shutil.copy(opdir / scfname, opdir / opfilename)
        txtline = line[1:]
        txtline.insert(0, opfilename)
        writeto.write(" ".join(txtline) + "\n")


def pairsubtraction(pair, group,
                    opdir, opf_prefix,
                    writeto,
                    fluxext=[0],
                    varext=None):
    """
    Perform subtraction between dither pairs of astronomical frames.

    For the given pair of dithers in a group, this function reads the
    corresponding dither text files, iterates over all combinations of
    frames, performs subtraction of one frame from the other, and writes a
    descriptive line for each output to the provided writable file.

    The output file names are constructed combining prefix, dither names,
    and indices if multiple frames are present.

    Args:
        pair (str): Two-character string denoting dither pair (e.g., 'AB').
        group (str/int): Identifier for frame group.
        opdir (Path): Path to the output directory containing dither files.
        opf_prefix (str): Prefix string used for naming output files.
        writeto (file-like): Open file handle for writing summary info.
        fluxext (list, optional): List indicating flux extensions to process
            (default: [0]).
        varext (optional): Variance extension to be processed (default: None).

    Writes:
        Lines describing each subtraction to `writeto` file.

    Calls:
        helper `operate_process` to perform the subtraction on frame
        file pairs.

    Notes:
        Assumes existence of helper function `read_dither_txtfile` to load
        dither frame info, and `operate_process` to compute subtraction.
    """
    logger = get_logger("dither")
    first_dithers = read_dither_txtfile(pair[0],
                                        group,
                                        opdir)
    second_dithers = read_dither_txtfile(pair[1],
                                         group,
                                         opdir)
    filename = opf_prefix
    # If there is more than one line
    # in each dither txtfile,
    # it should go through each combination
    # and do subtraction.
    # In readytoreduce, the lamps
    # of only first one will be written.
    for n, line_first in enumerate(first_dithers):
        frame_1 = line_first[0]
        filename = filename + "_" + pair[0]
        if len(first_dithers) > 1:
            filename += str(n)
        for m, line_second in enumerate(second_dithers):
            frame_2 = line_second[0]
            filename = filename + "-" + pair[1]
            if len(second_dithers) > 1:
                filename += str(m)
            filename += ".fits"
            txtline = line_first[1:]
            txtline.insert(0, filename)
            writeto.write(" ".join(txtline) + "\n")
            frame_1 = opdir / frame_1
            frame_2 = opdir / frame_2
            op_fname = opdir / filename
            logger.info("Dither subtraction")
            operate_process(frame_1, frame_2,
                            op_fname, '-',
                            fluxext=fluxext,
                            varext=varext)
            logger.info(f"{frame_1} - {frame_2} = {op_fname}")


def subtract_dithers(config, datadir):
    """
    Interactively process groups of dithered frames for subtraction or copying.

    This function iterates over groups of astronomical data, prompting the
    user to specify a filename prefix and pairs of dither instructions.
    For groups with a single dither, it copies the frame without subtraction.
    For each user-provided instruction: a single character copies one frame,
    while a two-character string indicates a pairwise subtraction operation
    (e.g., 'AB' subtracts B from A). The function writes the processed frame
    information to a summary text file for each group.

    Args:
        config (dict): Configuration dictionary with an 'outputs' key,
            which must contain 'OP_DIR' for specifying the output directory.
        datadir (str or Path): Name of the subdirectory containing the
            dataset to process.

    Prompts:
        - Output file prefix.
        - Space-separated dither pairs or single-frame instructions
          for each group (e.g., 'AB BA A').

    Writes:
        For each group, creates a text file 'ReadyToReduct_group<GROUP>.txt'
        in the output directory with records of operations performed.

    Notes:
        Single-letter instructions copy frames; two-letter instructions
        perform subtractions. Interactive input required.
    """
    logger = get_logger("dither")
    opdir = Path(config['outputs']['OP_DIR']) / datadir
    groups_dithers = get_dithers(opdir)

    print("\n")
    print("-" * 30)
    print("Enter the pairs to subtract in space separated form")
    print("For example an input: AB BA A")
    print(
        "Corresponding images produced by subtraction or not are :",
        "A-B, B-A and A"
    )
    print("Note: the final A is not a subtracted image")
    print("-" * 30)
    print("\n")
    for groups, dithers in groups_dithers.items():
        print("Running for group", groups)
        logger.info(f"Running for group {groups}")
        outfileprefix = get_filename(groups, opdir)
        dithers.sort()
        writeto = open(opdir / "ReadyToReduce_group{}.txt".format(groups), 'w')
        if config['inits']['TIMESERIES'] == 'Y':
            print("Timeseries data")
            for dith in dithers:
                instr = int(dith)
                opfname = outfileprefix + "_" + str(instr)
                copy_nopair_frames(instr,
                                   groups,
                                   opdir,
                                   opfname,
                                   writeto)

        else:
            if len(dithers) == 1:
                opfname = outfileprefix
                print("No dithers to subtract in this group")
                copy_nopair_frames(0, groups,
                                   opdir,
                                   opfname,
                                   writeto)
                logger.info("No dither frames")
                continue
            print("Doing for Group {}".format(groups))
            subpairs = input("Pairs to process:")
            subpairs = subpairs.split()
            logger.info(f"User entered {subpairs}")
            for instr in subpairs:
                logger.info(f"Running for pair: {instr}")
                if len(instr) == 1:
                    opfname = outfileprefix + "_" + instr
                    copy_nopair_frames(instr,
                                       groups,
                                       opdir,
                                       opfname,
                                       writeto)
                elif len(instr) == 2:
                    logger.info(f"Pair subtraction: {instr} {groups}")
                    pairsubtraction(instr, groups, opdir, outfileprefix,
                                    writeto)
        writeto.close()

# -----------------------------
# Imaging dithers
# -----------------------------


def select_reference_positions(dither_dict, opdir):
    """
    Interactively determine relative dither offsets from user-selected targets.

    Each frame corresponding to a dither position is displayed, and the user
    selects one or more reference targets. The centroid positions in the first
    frame are used as the reference, and the relative offsets of all
    subsequent frames are computed with respect to it.

    Parameters
    ----------
    dither_dict : dict
        Dictionary mapping dither identifiers to image filenames.
    opdir : str or pathlib.Path
        Directory containing the image files.

    Returns
    -------
    dict
        Dictionary mapping each dither identifier to a NumPy array
        ``[y_shift, x_shift]`` giving its offset relative to the reference
        frame. The reference frame has an offset of ``[0, 0]``.

    Notes
    -----
    If multiple targets are selected in a frame, the average offset of all
    selected targets is used.
    """
    print("Frames of each dither position will be displayed")
    print("Follow same order to select the target in all frames")
    difference_positions = {}
    reference = None
    for dither, fname in dither_dict.items():
        print(dither, fname)
        centroid = imageplot(opdir / fname,
                             title="{}:{}".format(dither, fname),
                             line_profile='aperture')
        if reference is None:
            reference = centroid
            difference_positions[dither] = np.array([0, 0])
        else:
            diff = centroid - reference
            if diff.shape[0] > 1:
                diff = np.mean(diff, axis=0)

            difference_positions[dither] = diff

    return difference_positions


def get_dither_shift_auto(dither_dict, opdir, config):
    """
    Automatically determine relative shifts between dithered images.

    The first image (or the image specified by ``REF_FRAME`` in the
    configuration) is used as the reference. The remaining images are aligned
    to the reference using phase cross-correlation.

    Parameters
    ----------
    dither_dict : dict
        Dictionary mapping dither identifiers to image filenames.
    opdir : str or pathlib.Path
        Directory containing the image files.
    config : dict
        Pipeline configuration dictionary. The reference image may be
        specified by ``config['dither']['REF_FRAME']``.

    Returns
    -------
    dict
        Dictionary mapping each dither identifier to a NumPy array
        ``[y_shift, x_shift]`` representing the measured image shift relative
        to the reference frame.
    """
    ref_image = opdir / config['dither']['REF_FRAME']
    if not ref_image.exists():
        ref_image = None
    shift_dict = {}
    for dither, fname in dither_dict.items():
        print(dither, fname)
        if ref_image is None:
            ref_image = opdir / fname
            shift_dict[dither] = np.array([0., 0.,])
        else:
            shifts = find_shift(ref_image, opdir / fname,
                                config)
            shift_dict[dither] = shifts[0]
    return shift_dict


def align_frames(dither_dict, shift_dict, opdir, config):
    """
    Apply measured shifts to a set of dithered images.

    Each image is shifted according to the corresponding value in
    ``shift_dict`` and written to a new FITS file prefixed with
    ``"Shifted_"``.

    Parameters
    ----------
    dither_dict : dict
        Dictionary mapping dither identifiers to image filenames.
    shift_dict : dict
        Dictionary mapping dither identifiers to image shifts as
        ``[y_shift, x_shift]``.
    opdir : str or pathlib.Path
        Directory containing the input images and where the shifted images
        will be written.
    config : dict
        Pipeline configuration containing the flux and variance FITS
        extensions.

    Returns
    -------
    list of pathlib.Path
        Paths to the shifted FITS images generated by this function.
    """
    alighed_fnames = []
    for dither, fname in dither_dict.items():
        opfname = "Shifted_" + fname
        opfname = opdir / opfname
        alighed_fnames.append(opfname)
        shifting_pos = shift_dict[dither]
        shifting_frame(opdir / fname,
                       opfname,
                       shifttoapply=shifting_pos,
                       fluxext=list(config['inputs']['FLUXEXT']),
                       varext=list(config['inputs']['VAREXT']))
    return alighed_fnames


def combine_dithers(config, datadir):
    """
    Combine dithered frames into a single science image and prepare it for
    spectral extraction.

    This function groups science frames according to their dither pattern.
    Groups containing multiple frames are aligned using either manually
    selected or automatically determined shifts before being combined into a
    single image. Groups containing only one frame are used directly without
    further processing.

    After the combined image is created, an interactive World Coordinate
    System (WCS) correction is performed. If a previously created list of
    target centroids and sky coordinates is available, it is reused;
    otherwise, the user is prompted to identify the targets, and a new list is
    generated. The user may edit this list before the WCS solution is applied.
    Finally, the filename of the processed image is written to a text file for
    use during spectral extraction.

    Parameters
    ----------
    config : configparser.ConfigParser
        Pipeline configuration containing the input/output directories,
        dither-combination parameters, and FITS extension information.
    datadir : str or pathlib.Path
        Relative path of the observation directory within the configured
        output directory.

    Notes
    -----
    - Frames are combined only when more than one dither position is present.
    - The frame alignment method is determined by the ``AUTODITHER``
      configuration option.
    - The WCS correction step is interactive and allows the user to review
      and modify the target list before applying the solution.
    - The final image filename is recorded in
      ``Readytoextract_group<group>.txt`` for use by the spectral extraction
      stage.
    """
    logger = get_logger("dither")
    opdir = Path(config['outputs']['OP_DIR']) / datadir
    groups_dithers = get_dithers(opdir, mode="P")
    print("\n")
    print("-" * 30)

    fluxext = list(config['inputs']['FLUXEXT'])
    varext = list(config['inputs']['VAREXT'])

    for groups in groups_dithers:
        print("Running for group", groups)
        logger.info(f"Running for group {groups}")
        outfileprefix = get_filename(groups, opdir)
        dither_dict = text_to_dict(
            txtfname=opdir / f"Clean_frame_group{groups}_dFull.txt"
        )

        if len(dither_dict) == 1:
            print("Single image. Nothing to combine")
            ditherkey = list(dither_dict.keys())[0]
            outfilename = opdir / dither_dict[ditherkey]
            logger.info(f"One frame. saved as {outfilename}")
        else:
            outfilename = opdir / f"{outfileprefix}.fits"
            print("outfilename", outfilename)

            if outfilename.exists():
                print(outfilename.name, "already exists. Skipping")
                continue

            if config['dither']['AUTODITHER'] == 'N':
                shift_dict = select_reference_positions(dither_dict, opdir)

            else:
                shift_dict = get_dither_shift_auto(dither_dict, opdir,
                                                   config)
            logger.info(f"Aligning {dither_dict}, {shift_dict}")
            aligned_fnames = align_frames(
                dither_dict, shift_dict, opdir,
                config)
            logger.info(f"Aligned combined as {aligned_fnames}")
            logger.info("Combining aligned frames")
            logger.info(f"Saving as {outfilename}")
            combine_process(aligned_fnames,
                            outfilename,
                            method='mean',
                            fluxext=fluxext,
                            varext=varext
                            )

        print("Running WCS correction")
        logger.info("WCS correction")
        config_wcs_fname = config['wcs']['WCS_POSITIONS']
        if len(config_wcs_fname) == 0:
            tar_wcs_fname_suggestion = f"{outfilename.stem}_wcstargets.txt"
            print(
                "If you have a list of WCS targets created",
                "in a previous trial,"
            )
            print("enter that filename here. Otherwise, press Enter.")
            tar_wcs_fname = input(
                "Enter the WCS list filename here:"
            ) or tar_wcs_fname_suggestion
            logger.info(f"User entered {tar_wcs_fname}")
        else:
            logger.info("Taking WCS initial condition from config")
            tar_wcs_fname = config_wcs_fname
        tar_wcs_fname = opdir / tar_wcs_fname
        if tar_wcs_fname.exists():
            print(tar_wcs_fname, "already exists.")
            print("Using that for WCS correction")
            logger.info(f"Using {tar_wcs_fname} for WCS correction.")
        else:
            centroids_list = imageplot(outfilename,
                                       title=outfilename,
                                       line_profile='aperture',
                                       get_target=True)
            headers = ['y_init', 'x_init',
                       'Target',
                       'RA', 'Dec',
                       'pmRA', 'pmDec']
            write_asciitable(centroids_list,
                             tar_wcs_fname,
                             headers=headers)
            logger.info(f"{tar_wcs_fname} saved for WCS correction")
        if config['wcs']['DISPLAY_FILE'] == 'T':
            logger.info("User like to see the text editor")
            logger.info("It was set in the config file")
            print("Opening the text editor with the target centroid")
            print("and their wcs information.")
            print("You can make changes in this if necessary.")
            print(tar_wcs_fname)
            open_in_editor(tar_wcs_fname, config)

        logger.info("Running WCS correction")
        wcs_correction(outfilename, tar_wcs_fname, config)

        finalframes_fname = opdir / f"Readytoextract_group{groups}.txt"
        with open(finalframes_fname, "w") as finalframes:
            finalframes.write(outfilename.name)

# End
