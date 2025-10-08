import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import read_fits_data
from .utils import fit_gaussian_profile


def imageplot(fname, ext=0, title=None, line_profile=False, **kwargs):
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
    if line_profile:
        enable_line_profile(fig, axs, data)
    plt.show()


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


def plot_main():
    parser = argparse.ArgumentParser(
        description="Plotting the fits frame")
    parser.add_argument("fname", type=str, help="Filename to display")
    parser.add_argument("--ext", type=int, help="Extension to display",
                        default=0)
    args = parser.parse_args()
    fname = args.fname
    imageplot(fname, title="file : " + fname, line_profile=True)

# End
