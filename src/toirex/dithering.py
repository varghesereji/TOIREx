#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
from scipy.ndimage import median_filter
from skimage import registration
import shutil

from ariastrotools import operate_process

from .utils import read_fits_data
from .utils import text_to_dict
from .utils import extract_number_from_fname
from .utils import read_txt_file
from .utils import get_filename

from .plottings import imageplot

# ----------------#
# Identify dither #
# ----------------#


def filter_image(frame, size=(20, 20)):
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
        Default is (20, 20).

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


def find_shift(frame_1, frame_2, config):
    """
    Compute the relative shift between two FITS images.

    This function reads two FITS frames, crops them based on configuration
    settings, applies filtering, and uses phase cross-correlation to
    determine the translation required to align the second frame with the
    first.

    Parameters
    ----------
    frame_1 : str or Path
        Path to the first FITS file (reference image).
    frame_2 : str or Path
        Path to the second FITS file (moving image).
    config : dict
        Configuration dictionary containing cropping settings under
        ``config['dither']['CROP']``. The crop should be a space-separated
        string of four integers representing:
        [crop_y_bottom crop_y_top crop_x_left crop_x_right].
    upsample_factor : int, optional
        Upsampling factor for subpixel accuracy in the phase
        cross-correlation (default is 50).

    Returns
    -------
    tuple
        A tuple containing:
        - shift : ndarray
            Pixel shift needed to align `frame_2` to `frame_1`.
        - error : float
            Normalized root-mean-square error after alignment.
        - diffphase : float
            Global phase difference between the two images in radians.

    Example
    -------
    >>> config = {'dither': {'CROP': "10 100 20 200"}}
    >>> shift_info = find_shift("image1.fits", "image2.fits", config)
    >>> print(shift_info)
    (array([dy, dx]), error_value, diffphase_value)
    """

    frame_1 = read_fits_data(frame_1)
    frame_2 = read_fits_data(frame_2)
    crop = config['dither']['CROP']
    upsample_factor = int(config['dither']['UPSAMPLE'])
    crop = [int(x) for x in crop.strip().split(" ")]
    crop_yb = crop[0]
    crop_yt = crop[1]
    crop_xl = crop[2]
    crop_xr = crop[3]
    filt_img1 = filter_image(frame_1[crop_yb:crop_yt, crop_xl:crop_xr])
    filt_img2 = filter_image(frame_2[crop_yb:crop_yt, crop_xl:crop_xr])
    difference = registration.phase_cross_correlation(
        reference_image=filt_img1,
        moving_image=filt_img2,
        upsample_factor=upsample_factor)
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
        # The text file name have two integers. So, the
        # function will return two numners. first one
        # will be the group number, and second one
        # will be dither number.
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
            operate_process(frame_1, frame_2,
                            op_fname, '-',
                            fluxext=fluxext,
                            varext=varext)


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
        outfileprefix = get_filename(groups, opdir)
        dithers.sort()  # Just making them to be ascending order
        writeto = open(opdir / "ReadyToReduce_group{}.txt".format(groups), 'w')

        if len(dithers) == 1:
            opfname = outfileprefix
            print("No dithers to subtract in this group")
            copy_nopair_frames(0, groups,
                               opdir,
                               opfname,
                               writeto)
            continue
        print("Doing for Group {}".format(groups))
        subpairs = input("Pairs to process:")
        subpairs = subpairs.split()

        for instr in subpairs:
            if len(instr) == 1:
                opfname = outfileprefix + "_" + instr
                copy_nopair_frames(instr,
                                   groups,
                                   opdir,
                                   opfname,
                                   writeto)
            elif len(instr) == 2:
                pairsubtraction(instr, groups, opdir, outfileprefix, writeto)
        writeto.close()

# -----------------------------
# Imaging dithers
# -----------------------------


def select_reference_positions(dither_dict, opdir):
    print("Frames of each dither position will be displayed")
    print("Select at least two targes in each frame")
    print("Follow same order to select the target in all frames")
    selected_positions = {}
    for dither, fname in dither_dict.items():
        print(dither, fname)
        centroid = imageplot(opdir / fname,
                             title="{}:{}".format(dither, fname),
                             line_profile='aperture')
        selected_positions[dither] = centroid
    return selected_positions


def combine_dithers(config, datadir):
    opdir = Path(config['outputs']['OP_DIR']) / datadir
    groups_dithers = get_dithers(opdir, mode="P")
    # print(groups_dithers)
    print("\n")
    print("-" * 30)
    for groups in groups_dithers:
        print("Running for group", groups)
        outfileprefix = get_filename(groups, opdir)
        print(groups, outfileprefix)
        dither_dict = text_to_dict(
            txtfname=opdir / "Clean_frame_group{}_dFull.txt".format(groups)
        )
        if len(list(dither_dict.keys())) == 1:
            print("Single image. Nothing to combine")
        else:
            select_reference_positions(dither_dict, opdir)
            



# End
