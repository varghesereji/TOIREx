import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
from astropy.wcs import WCS

from matplotlib.widgets import Slider, RadioButtons
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval
from astropy.visualization import LinearStretch
from astropy.visualization import LogStretch
from astropy.visualization import SqrtStretch


from .utils import read_fits_data
from .utils import read_fits_header
from .utils import fit_gaussian_profile
from .image_utils import select_source
from .io_utils import launch_simbad_gui


def imageplot(fname, ext=0, title=None, line_profile='drawline',
              get_target=False, centroid_list=None, **kwargs):
    data = read_fits_data(fname, ext=ext)

    header = read_fits_header(fname, ext=ext)
    wcs = WCS(header)
    if wcs.is_celestial:
        use_wcs = True
    else:
        use_wcs = False

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.mean(data) - 2.0 * (np.std(data))
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.mean(data) + 2.0 * (np.std(data))
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"
    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"

    # --- Initial normalization ---
    interval = ZScaleInterval()
    # stretch = LinearStretch()
    # norm = ImageNormalize(data, interval=interval, stretch=stretch)
    fig = plt.figure(figsize=(9, 9))
    if use_wcs:
        axs = fig.add_subplot(111, projection=wcs)
        axs.coords[0].set_axislabel('RA')
        axs.coords[1].set_axislabel('Dec')
    else:
        # fig, axs = plt.subplots(figsize=(8, 8))
        axs = fig.add_subplot(111)
    plt.subplots_adjust(left=0.55, bottom=0.05)
    im = axs.imshow(data, **kwargs)

    if title is not None:
        axs.set_title(title, loc="left")
    # --- Sliders for vmin/vmax ---
    ax_vmin = plt.axes([0.15, 0.05, 0.55, 0.03])
    ax_vmax = plt.axes([0.15, 0.0, 0.55, 0.03])
    s_vmin = Slider(ax_vmin, 'vmin', np.nanmin(data), np.nanmax(data),
                    valinit=np.nanmin(data))
    s_vmin = Slider(ax_vmin, 'vmin', kwargs['vmin'], kwargs['vmax'],
                    valinit=kwargs['vmin'])
    s_vmax = Slider(ax_vmax, 'vmax', kwargs['vmin'], kwargs['vmax'],
                    valinit=kwargs['vmax'])

    fig.canvas.draw_idle()
    # --- Radio buttons for stretch ---
    ax_stretch = plt.axes([0.005, 0.55, 0.10, 0.20])
    stretch_buttons = RadioButtons(ax_stretch, ('linear', 'sqrt', 'log'))

    # --- Radio buttons for colormap ---
    ax_cmap = plt.axes([0.005, 0.25, 0.10, 0.20])
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

        # Apply normalization and redraw
        norm = ImageNormalize(data, interval=interval, stretch=stretch,
                              vmin=s_vmin.val, vmax=s_vmax.val)
        im.set_norm(norm)
        im.set_cmap(cmap)
    # --- Connect the widgets ---
    for w in (s_vmin, s_vmax):
        w.on_changed(update)
    for w in (stretch_buttons, cmap_buttons):
        w.on_clicked(update)
    # stretch_buttons.on_clicked(update)
    # cmap_buttons.on_clicked(update)

    plt.tight_layout()
    fig.colorbar(im, ax=axs, label="Counts")
    # centroid_list = None
    if line_profile == 'drawline':
        enable_line_profile(fig, axs, data)
    elif line_profile == 'aperture':
        if isinstance(title, Path):
            title = title.name
        title = title + "\n Press Ctrl and click on apertures to select them"
        axs.set_title(title, loc="left")
        centroid_list = select_aperture(fig, axs, data, get_target,
                                        centroids_list=centroid_list)
    plt.show()
    return np.array(centroid_list)


def enable_line_profile(fig, ax, image):
    line_coords = []
    # plt.show()
    # annotation = None

    def onclick(event):
        nonlocal line_coords
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


def select_aperture(fig, ax, image, get_target=False, centroids_list=None):
    if centroids_list is None:
        centroids_list = []
    circles_list = []

    for c in centroids_list:
        y_center, x_center = c[:2]
        print(y_center, x_center)
        circle = Circle((x_center, y_center), 10,
                        edgecolor='red',
                        facecolor='none',
                        linewidth=2)
        ax.add_patch(circle)
        circles_list.append(circle)
    fig.canvas.draw_idle()
    # plt.show()
    # annotation = None

    def onclick(event):
        # nonlocal line_coords
        if event.inaxes != ax:
            return
        if event.key not in ['control', 'shift']:
            return  # quietly ignore other clicks

        xdata, ydata = event.xdata, event.ydata
        if event.key == 'control':
            sel_reg = image[int(ydata)-10:int(ydata)+10,
                            int(xdata)-10:int(xdata)+10]
            centroid = select_source(sel_reg)
            x_center = centroid[1] + xdata-10
            y_center = centroid[0] + ydata-10
            if not get_target:
                centroids_list.append([y_center,
                                       x_center])

            radius = 10
            circle = Circle((x_center, y_center), radius,
                            edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(circle)
            circles_list.append(circle)
            # fig.canvas.draw()
            # target_name = input("Enter target name")
            # query_object((xdata, ydata))
            if get_target:
                coords = launch_simbad_gui()
                target_coords = [int(y_center), int(x_center)]
                target_coords.append(coords['name'])
                target_coords.append(coords['ra'])
                target_coords.append(coords['dec'])
                target_coords.append(coords['pmra'])
                target_coords.append(coords['pmdec'])
                centroids_list.append(target_coords)
        elif event.key == 'shift':
            if not centroids_list:
                return
            points = np.array(centroids_list)[:, :2]
            distances = np.sqrt(
                (points[:, 0] - ydata)**2 +
                (points[:, 1] - xdata)**2
                )
            idx = np.argmin(distances)
            centroids_list.pop(idx)
            circles_list.pop(idx).remove()
        fig.canvas.draw_idle()
        # print(coords)
        # print(centroids_list)
    fig.canvas.mpl_connect("button_press_event", onclick)
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
