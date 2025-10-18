import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle

from .utils import read_fits_data
from .utils import fit_gaussian_profile
from .image_utils import select_source
from .io_utils import launch_simbad_gui


def imageplot(fname, ext=0, title=None, line_profile='drawline',
              get_target=False, **kwargs):
    data = read_fits_data(fname, ext=ext)

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.mean(data) - 1.0 * (np.std(data))
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.mean(data) + 1.0 * (np.std(data))
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"
    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"
    fig, axs = plt.subplots(figsize=(9, 9))
    im = axs.imshow(data, **kwargs)
    if title is not None:
        axs.set_title(title, loc="left")
    plt.tight_layout()
    fig.colorbar(im, ax=axs, label="Counts")
    centroid_list = None
    if line_profile == 'drawline':
        enable_line_profile(fig, axs, data)
    elif line_profile == 'aperture':
        if isinstance(title, Path):
            title = title.name
        title = title + "\n Press Ctrl and click on apertures to select them"
        axs.set_title(title, loc="left")
        centroid_list = select_aperture(fig, axs, data, get_target)
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


def select_aperture(fig, ax, image, get_target=False):
    centroids_list = []
    # plt.show()
    # annotation = None

    def onclick(event):
        # nonlocal line_coords
        if event.inaxes != ax:
            return
        if not (event.key == 'control'):
            return  # quietly ignore other clicks

        xdata, ydata = event.xdata, event.ydata
        # ax.plot(xdata, ydata, 'o', color='red')
        # print(xdata, ydata)
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
        fig.canvas.draw()
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
