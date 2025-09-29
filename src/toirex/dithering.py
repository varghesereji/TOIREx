#!/usr/bin/env python3

from scipy.ndimage import median_filter
from skimage import registration


from .utils import read_fits_data


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


# def DitherDetection(ObjectFile, ContWindowSelection,
#                     startLoc=None, avgHWindow=21, TraceHWidth=5):

#     """identify the center of a spectrum window """
#     if isinstance(ObjectFile, str):
#         ObjectFile = read_fits_data(ObjectFile)

#     if startLoc is None:
#         startLoc = ObjectFile.shape[1]//2
#     # Starting labelling Reference XD cut data;
#     WindowStart = ContWindowSelection[0]
#     WindowEnd = ContWindowSelection[1]
#     RefXD = np.nanmedian(ObjectFile[WindowStart:WindowEnd,
#                                     startLoc-avgHWindow:startLoc+avgHWindow],
#                          axis=1)
#     Refpixels = np.arange(len(RefXD))+WindowStart
#     Bkg = signal.order_filter(
#         RefXD, domain=[True]*TraceHWidth*5, rank=int(TraceHWidth*5/10)
#     )
#     Flux = np.abs(RefXD - Bkg)
#     ThreshMask = RefXD > (Bkg + np.abs(mad_std(Flux))*6)
#     centerpix = np.sum(
#         Flux[ThreshMask]*Refpixels[ThreshMask]
#     ) / np.sum(Flux[ThreshMask])

#     return centerpix

# End
