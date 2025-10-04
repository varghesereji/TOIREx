#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
from scipy.ndimage import median_filter
from skimage import registration
import shutil

from ariastro import operate_process

from .utils import read_fits_data
from .utils import extract_number_from_fname
from .utils import read_txt_file

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
    groups_dithers_dict = defaultdict(list)
    for sublist in groups_dithers_list:
        groups_dithers_dict[sublist[0]].append(sublist[1])
    return groups_dithers_dict


def get_dithers(opdir):
    dithertxt_fname = "Clean_frame_group*_d*.txt"
    dither_txtfiles = list(opdir.glob(dithertxt_fname))
    groups_dithers = []
    for group in dither_txtfiles:
        # The text file name have two integers. So, the
        # function will return two numners. first one
        # will be the group number, and second one
        # will be dither number
        numbers = extract_number_from_fname(group.name)
        groups_dithers.append(numbers)
    return dithers_and_groups(groups_dithers)


# ----------------------------- #
# Subtract dithers in spectra   #
# ----------------------------- #


def read_dither_txtfile(pairstr, group,
                        opdir):
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
    opdir = Path(config['outputs']['OP_DIR']) / datadir
    groups_dithers = get_dithers(opdir)
    outfileprefix = input(
        "Enter the prefix of you want for reduce 1d spectra:"
    )
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
        dithers.sort()  # Just making them to be ascending order
        writeto = open(opdir / "ReadyToReduct_group{}.txt".format(groups), 'w')
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

# End
