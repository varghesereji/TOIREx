import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from matplotlib.patches import Circle
from astropy.wcs import WCS

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.widgets import RadioButtons
from matplotlib.widgets import RangeSlider
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval
from astropy.visualization import LinearStretch
from astropy.visualization import LogStretch
from astropy.visualization import SqrtStretch


from .utils import read_fits_data
from .utils import read_fits_header
from .utils import fit_gaussian_profile
from .image_utils import select_source
from .image_utils import get_radial_profile
from .image_utils import make_cutout
from .io_utils import launch_simbad_gui


def plot_epsf(epsf, fitted_stars, plot_fname="epsf_plot.pdf",
              show_plot=True):
    """
    Create diagnostic plots for the generated effective PSF (ePSF).

    This function generates a two-page PDF containing the constructed ePSF
    and the stellar cutouts used to build it. Optionally, the figures can
    also be displayed interactively.

    Parameters
    ----------
    epsf : `photutils.psf.ImagePSF` or `photutils.psf.EPSFModel`
        Effective PSF model produced by the ePSF builder.
    fitted_stars : `photutils.psf.EPSFStars`
        Collection of fitted stellar cutouts used to construct the ePSF.
    plot_fname : str or pathlib.Path, optional
        Full path and filename for the output PDF. If the filename does not
        have a ``.pdf`` extension (case-insensitive), it is added
        automatically. Default is ``"epsf_plot.pdf"``.
    show_plot : bool, optional
        If `True`, display the generated figures after saving them to the
        PDF. If `False`, close the figures without displaying them.
        Default is `True`.

    Notes
    -----
    The generated PDF contains two pages:

    1. The effective PSF image.
    2. The stellar cutouts used to construct the ePSF, annotated with their
       index and fitted center coordinates.
    """
    plot_fname = Path(plot_fname)
    if plot_fname.suffix.lower() != ".pdf":
        plot_fname = plot_fname.with_suffix(".pdf")

    pdf = PdfPages(plot_fname)

    # Plotting epsf, page 1
    fig1, ax = plt.subplots(figsize=(12, 12))
    axim = ax.imshow(epsf.data, origin="lower")
    fig1.colorbar(axim)
    fig1.suptitle("Effective PSF")
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig1, bbox_inches="tight")

    # Plotting stars used
    n = len(fitted_stars)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(2*ncols, 2*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, star) in enumerate(zip(axes, fitted_stars), start=1):
        ax.imshow(star.data, origin="lower", cmap="gray")
        x, y = star.center
        ax.set_title(f"{i}\n{x:.1f},{y:.1f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for x in axes[n:]:
        x.axis("off")
    fig2.suptitle("Stars used to make ePSF")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig2, bbox_inches="tight")
    pdf.close()

    if show_plot:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    print("ePSF saved as:", plot_fname)


def save_residualimg(data, residual, fname="Residual_plot.pdf",
                     show_plot=True):
    """
    Save a comparison plot of the data, model, and residual image.

    The figure contains three panels showing the original data, the fitted
    model (computed as ``data - residue``), and the residual image. All
    panels are displayed using the same intensity scale for direct
    comparison, with a shared colourbar.

    Parameters
    ----------
    data : ndarray
        Original image data.
    residual : ndarray
        Residual image, defined as the difference between the data and the
        fitted model.
    fname : str or pathlib.Path, optional
        Filename for the output figure. The default is
        ``"Residue_plot.pdf"``.
    show_plot : bool, optional
        If `True`, display the figure after saving. Otherwise, close the
        figure without displaying it.

    Returns
    -------
    None
    """
    vmin = np.nanmean(data) - 2.0 * np.nanstd(data)
    vmax = np.nanmean(data) + 2.0 * np.nanstd(data)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    fig.subplots_adjust(right=0.88)
    im = ax[0].imshow(data, vmin=vmin, vmax=vmax, origin='lower')
    ax[1].imshow(data - residual, vmin=vmin, vmax=vmax, origin='lower')
    ax[2].imshow(residual, vmin=vmin, vmax=vmax, origin='lower')
    ax[0].set_title("Data")
    ax[1].set_title("Model")
    ax[2].set_title("Residual Image")

    # Create a dedicated axis for the colorbar
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cax)

    # fig.tight_layout()
    plt.savefig(fname)
    print(f"Saving the residual image as {fname}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def imageplot(fname, ext=0, title=None, line_profile='drawline',
              get_target=False, centroid_list=None,
              aperture_radii=(10, 15, 20),
              save_plot=None,
              show_plot=True,
              **kwargs):
    """
    Display a FITS image with interactive visualization tools.

    The displayed image includes controls for adjusting the intensity
    range, image stretch, and colormap. Depending on the selected
    interactive mode, line profiles can be drawn or source apertures can
    be selected directly on the image.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the FITS image.
    ext : int, optional
        FITS extension containing the image data. Default is 0.
    title : str or pathlib.Path, optional
        Title displayed above the image. If a `Path` object is supplied,
        only its filename is displayed.
    line_profile : {'drawline', 'aperture'}, optional
        Interactive mode to enable. ``'drawline'`` enables interactive
        line-profile measurements, while ``'aperture'`` enables
        interactive aperture selection. Default is ``'drawline'``.
    get_target : bool, optional
        If `True`, query SIMBAD for target information after selecting a
        source in aperture mode. Default is `False`.
    centroid_list : list, optional
        Initial list of source centroids. Existing apertures are
        displayed when aperture mode is enabled. If `None`, an empty list
        is used.
    aperture_radii : tuple of float, optional
        Tuple specifying the source aperture radius, background inner
        radius, and background outer radius as
        ``(source_radius, bkg_inner, bkg_outer)``. Used only when
        ``line_profile='aperture'``. Default is ``(10, 15, 20)``.
    save_plot : str or pathlib.Path, optional
        Filename to save the displayed figure. If `None`, the figure is
        not saved. Default is `None`.
    show_plot : bool, optional
        If `True`, display the detected sources overlaid on the image.
        Default is `True`.

    **kwargs
        Additional keyword arguments passed to
        `matplotlib.axes.Axes.imshow`.

    Returns
    -------
    ndarray
        Array containing the selected source centroids. If
        ``get_target=True``, additional target information returned by
        the SIMBAD query is included for each selected source.

    Raises
    ------
    ValueError
        If ``aperture_radii`` does not contain exactly three values or if
        the radii do not satisfy
        ``source_radius < bkg_inner < bkg_outer``.

    Notes
    -----
    In aperture mode, the following mouse interactions are available:

    - **Ctrl + Left Click** : Add a source after centroid refinement.
    - **Shift + Left Click** : Remove the nearest selected source.
    """
    data = read_fits_data(fname, ext=ext)

    header = read_fits_header(fname, ext=ext)
    wcs = WCS(header)
    if wcs.is_celestial:
        use_wcs = True
    else:
        use_wcs = False

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.nanmean(data) - 2.0 * (np.nanstd(data))
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.nanmean(data) + 2.0 * (np.nanstd(data))
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"
    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"

    # --- Initial normalization ---
    interval = ZScaleInterval()

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(
        nrows=1,
        ncols=3,
        width_ratios=[1.2, 6, 0.25],  # controls | image | colorbar
        wspace=0.05,
        figure=fig
        )
    if use_wcs:
        axs = fig.add_subplot(gs[0, 1], projection=wcs)
        axs.coords[0].set_axislabel('RA')
        axs.coords[1].set_axislabel('Dec')
    else:

        axs = fig.add_subplot(gs[0, 1])

    im = axs.imshow(data, **kwargs)

    if title is not None:
        axs.set_title(title, loc="left")

    # --- Sliders for vmin/vmax ---
    ax_range = plt.axes([0.25, 0.05, 0.5, 0.035])
    s_range = RangeSlider(
        ax=ax_range,
        label='vmin/vmax',
        valmin=np.nanpercentile(data, 1),
        valmax=np.nanpercentile(data, 99.9),
        valinit=(kwargs['vmin'], kwargs['vmax']),
        )
    fig.canvas.draw_idle()

    # --- Radio buttons for stretch ---
    ax_stretch = fig.add_axes([0.005, 0.55, 0.12, 0.25])
    stretch_buttons = RadioButtons(ax_stretch, ('linear', 'sqrt', 'log'))

    # --- Radio buttons for colormap ---
    ax_cmap = fig.add_axes([0.005, 0.25, 0.12, 0.25])
    cmap_buttons = RadioButtons(ax_cmap, ('gray', 'viridis', 'inferno'))

    # --- Update function ---
    def update(_):
        # Stretch type
        stretch_type = stretch_buttons.value_selected
        stretch = {
            'linear': LinearStretch(),
            'sqrt': SqrtStretch(),
            'log': LogStretch()
        }[stretch_type]

        # Colormap
        cmap = cmap_buttons.value_selected
        vmin, vmax = s_range.val
        # Apply normalization and redraw
        norm = ImageNormalize(data, interval=interval, stretch=stretch,
                              vmin=vmin, vmax=vmax)
        im.set_norm(norm)
        im.set_cmap(cmap)

    # --- Connect the widgets ---
    s_range.on_changed(update)

    plt.tight_layout()

    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Counts")
    # centroid_list = None
    if line_profile == 'drawline':
        enable_line_profile(fig, axs, data)
    elif line_profile == 'aperture':
        if isinstance(title, Path):
            title = title.name
        title = title + "\n Press Ctrl and click on apertures to select them"
        title += "\n Press Shift and click to remove selected apertures"
        axs.set_title(title, loc="left")
        if len(aperture_radii) != 3:

            raise ValueError(
                "'aperture_radii' must contain "
                "(source_radius, bkg_inner, bkg_outer)."
            )

        radius, bkg_in, bkg_out = aperture_radii

        if not (radius < bkg_in < bkg_out):
            raise ValueError(
                "Expected source_radius < bkg_inner < bkg_outer."
            )

        centroid_list = select_aperture(fig, axs, data,
                                        radius=radius,
                                        bkgs=(bkg_in, bkg_out),
                                        get_target=get_target,
                                        centroids_list=centroid_list)

    if show_plot:
        plt.show()
    if save_plot is not None:
        print("Plot saved", save_plot)
        fig.savefig(save_plot)

    if not show_plot:
        plt.close(fig)

    return np.array(centroid_list)


def enable_line_profile(fig, ax, image):
    line_coords = []
    # plt.show()
    # annotation = None

    def onclick(event):
        nonlocal line_coords

        toolbar = fig.canvas.toolbar

        if toolbar.mode != '':
            return

        if event.inaxes != ax:
            return
        xdata, ydata = event.xdata, event.ydata
        line_coords.append((xdata, ydata))
        # if len(line_coords) == 1:
        # ax.plot(xdata, ydata, 'o', color='red')
        if len(line_coords) == 2:
            (x0, y0), (x1, y1) = line_coords

            # Draw the ine
            ax.plot([x0, x1], [y0, y1], color='red')
            fig.canvas.draw()

            # Sample pixels along the line
            length = int(np.hypot(x1-x0, y1-y0))
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
            counts = image[y.astype(int), x.astype(int)]  # nearest neighbor
            x, counts, fitted_counts, g_fit = fit_gaussian_profile(counts)
            # Show counts plot
            plt.figure()
            plt.plot(x, counts, label="Profile")
            plt.plot(x, fitted_counts, label="Gaussian fit", linestyle='--')
            plt.xlabel("Pixel index along line")
            plt.ylabel("Counts")
            plt.title("Counts along line")
            plt.annotate(
                f"fwhm : {2.355*g_fit.stddev.value:.3f}",
                xy=(0, 1),
                xycoords='axes fraction',
                xytext=(10, 10),
                textcoords='offset points',
                color='black',
                fontsize=9,
                bbox=dict(boxstyle='round, pad=0.3', fc='white', alpha=0.5)
                )

            plt.show()

            line_coords = []  # Reset for next line

    fig.canvas.mpl_connect("button_press_event", onclick)


def mark_source(ax, center,
                radius=10,
                bkgs=(15, 20)):
    """
    Draw the source aperture and background annulus on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the apertures are drawn.
    center : tuple of float
        ``(x, y)`` coordinates of the source centre in pixel units.
    radius : float, optional
        Radius of the source aperture in pixels. Default is 10.
    bkgs : tuple of float, optional
        Inner and outer radii of the background annulus in pixels, given
        as ``(inner_radius, outer_radius)``. Default is ``(15, 20)``.

    Returns
    -------
    source : matplotlib.patches.Circle
        Circle representing the source aperture.
    bkg_in : matplotlib.patches.Circle
        Circle representing the inner boundary of the background annulus.
    bkg_out : matplotlib.patches.Circle
        Circle representing the outer boundary of the background annulus.

    Raises
    ------
    ValueError
        If ``bkgs`` does not contain exactly two radii or if the inner
        radius is greater than or equal to the outer radius.
    """

    common = dict(facecolor="none", linewidth=2)

    # source aperture
    source = Circle(center,
                    radius=radius,
                    edgecolor='red',
                    **common)

    # bkg circles
    if len(bkgs) != 2:
        raise ValueError(
            "'bkgs' must contain (inner_radius, outer_radius)."
        )

    if bkgs[0] >= bkgs[1]:
        raise ValueError(
            "Background inner radius must be smaller than outer radius."
        )

    bkg_in = Circle(center, bkgs[0],
                    edgecolor='green',
                    **common)

    bkg_out = Circle(center, bkgs[1],
                     edgecolor='green',
                     **common)

    for patch in (source, bkg_in, bkg_out):
        ax.add_patch(patch)

    return source, bkg_in, bkg_out


def select_aperture(fig,
                    ax,
                    image,
                    radius=10,
                    bkgs=(15, 20),
                    get_target=False,
                    centroids_list=None):
    """
    Interactively select source apertures on an image.

    Existing sources are displayed using a circular source aperture and
    background annulus. Additional sources can be selected by holding the
    Ctrl key and clicking near a source, while the nearest selected source
    can be removed by holding the Shift key and clicking near it. The
    clicked position is refined by centroiding within a small region
    around the click location.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the displayed image.
    ax : matplotlib.axes.Axes
        Axes on which the image is displayed.
    image : ndarray
        Two-dimensional image array.
    radius : float, optional
        Radius of the source aperture in pixels. Default is 10.
    bkgs : tuple of float, optional
        Inner and outer radii of the background annulus in pixels, given
        as ``(inner_radius, outer_radius)``. Default is ``(15, 20)``.
    get_target : bool, optional
        If `True`, prompt for SIMBAD target information after selecting a
        source and store the returned metadata together with the source
        coordinates. Default is `False`.
    centroids_list : list, optional
        Initial list of source centroids. Each element must contain at
        least the source coordinates as ``[y, x]``. If `None`, an empty
        list is created.

    Returns
    -------
    list
        Updated list of source centroids. If ``get_target=True``, each
        entry additionally contains the target name, equatorial
        coordinates, and proper motions returned by the SIMBAD query.

    Notes
    -----
    The following mouse interactions are supported:

    - **Ctrl + Left Click**: Add a source after centroid refinement.
    - **Shift + Left Click**: Remove the nearest selected source.

    The display is updated interactively as sources are added or removed.
    """
    if centroids_list is None:
        centroids_list = []

    aperture_patches = []

    # Draw existing apertures
    for c in centroids_list:
        y_center, x_center = c[:2]

        source, bkg_in, bkg_out = mark_source(
            ax,
            (x_center, y_center),
            radius=radius,
            bkgs=bkgs
        )

        aperture_patches.append(
            (source, bkg_in, bkg_out)
        )

    fig.canvas.draw_idle()

    def add_source(xdata, ydata):
        """Add a source near the clicked position"""

        half_box = 10

        y0 = max(0, int(ydata) - half_box)
        y1 = min(image.shape[0], int(ydata) + half_box)
        x0 = max(0, int(xdata) - half_box)
        x1 = min(image.shape[1], int(xdata) + half_box)
        sel_reg = image[y0:y1, x0:x1]

        centroid = select_source(sel_reg)

        x_center = centroid[1] + x0
        y_center = centroid[0] + y0

        source, bkg_in, bkg_out = mark_source(
            ax,
            (x_center, y_center),
            radius=radius,
            bkgs=bkgs
        )

        aperture_patches.append(
            (source, bkg_in, bkg_out)
        )

        if get_target:
            coords = launch_simbad_gui()
            centroids_list.append([
                int(y_center),
                int(x_center),
                coords["name"],
                coords["ra"],
                coords["dec"],
                coords["pmra"],
                coords["pmdec"],
            ])
        else:
            centroids_list.append(
                [
                    y_center,
                    x_center
                ]
            )

    def remove_source(xdata, ydata):
        """Remove the nearest selected source."""

        if not centroids_list:
            return
        points = np.asarray(centroids_list)[:, :2]

        distances = np.sqrt(
            (points[:, 0] - ydata)**2 +
            (points[:, 1] - xdata)**2
        )
        idx = np.argmin(distances)
        centroids_list.pop(idx)
        # aperture_patches.pop(idx).remove()
        source, bkg_in, bkg_out = aperture_patches.pop(idx)

        for patch in (source, bkg_in, bkg_out):
            patch.remove()

    def show_profile():
        """Display the radial profile of the most recently selected source."""

        if not centroids_list:
            print("No source selected.")
            return

        y_center, x_center = centroids_list[-1][:2]

        edge_radii = np.arange(bkgs[1] + 10)

        rp = get_radial_profile(
            image,
            (x_center, y_center),
            edge_radii=edge_radii,
        )

        fig_profile, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax_profile = ax[0]

        rp.plot(ax=ax_profile, label="Radial Profile")
        rp.plot_error(ax=ax_profile)

        ax_profile.plot(rp.radius,
                        rp.gaussian_profile,
                        label=f"Gaussian Fit\n gFWHM={rp.gaussian_fwhm:.3f}")
        ax_profile.plot(rp.radius,
                        rp.moffat_profile,
                        label=f"Moffat Fit\n mFWHM={rp.moffat_fwhm:.3f}")

        ax_profile.grid(alpha=0.3)

        ax_cutout = ax[1]
        cutout_size = int(2 * (bkgs[1] + 10))
        cutout = make_cutout(
            image,
            (x_center, y_center),
            (cutout_size, cutout_size)
        )
        ax_cutout.imshow(cutout.data, origin='lower')

        ax_profile.legend()
        plt.tight_layout()
        plt.show()

    def onclick(event):
        # nonlocal line_coords
        toolbar = fig.canvas.toolbar

        if toolbar.mode != '':
            return

        if event.inaxes != ax:
            return

        if event.key == "control":
            add_source(event.xdata, event.ydata)

        elif event.key == "shift":
            remove_source(event.xdata, event.ydata)

        else:
            return

        fig.canvas.draw_idle()

    def onkeypress(event):
        """Handle keyboard shortcuts."""

        toolbar = fig.canvas.toolbar

        if toolbar.mode != "":
            return

        if event.key is None:
            return

        if event.key.lower() == "r":
            show_profile()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkeypress)

    return centroids_list


def specplot(fname, ext=0, wlext=1, title=None, **kwargs):
    flux = read_fits_data(fname, ext=ext)
    wl = read_fits_data(fname, ext=wlext)
    plot_arrays(wl, flux, title=title, **kwargs)


def plot_arrays(wl_array, flux_array, title=None,
                fig_axs=None, clear=False, show=True,
                **kwargs):
    n = len(wl_array)
    ncols = 1 if n <= 5 else 2
    nrows = int(np.ceil(n / ncols))
    if fig_axs is None:
        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        axs = np.atleast_1d(axs).ravel()  # flatten even if it's 1D
    else:
        fig, axs = fig_axs
        if clear:
            for ax in axs:
                ax.clear()

    for i, wlarr in enumerate(wl_array):
        axs[i].plot(wlarr, flux_array[i], **kwargs)
        axs[i].set_ylim(bottom=-5)
        axs[i].legend()
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig, axs


def plot_main():
    parser = argparse.ArgumentParser(
        description="Plotting the fits frame")
    parser.add_argument("fname", type=str, help="Filename to display")
    parser.add_argument("--ext", type=int, help="Extension to display",
                        default=0)
    parser.add_argument("--xext",
                        type=int,
                        help="Extension to be added in x axis. \
                        If None, make imshow.",
                        default=None)
    parser.add_argument("--aperture",
                        type=str,
                        help="Draw line and fit gaussian or select sircular \
                        aperture (drawline, aperture)",
                        default='drawline')
    args = parser.parse_args()
    fname = args.fname
    if args.xext is not None:
        specplot(fname, args.ext, args.xext, title="file : " + fname)
    else:
        imageplot(fname, title="file : " + fname, line_profile=args.aperture)


if __name__ == "__main__":
    plot_main()

# End
