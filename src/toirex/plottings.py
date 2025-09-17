import numpy as np
import matplotlib.pyplot as plt

from .utils import read_fits_data


def imageplot(fname, title=None, **kwargs):
    data = read_fits_data(fname, ext=0)

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.mean(data) - 1.0 * (np.std(data))
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.mean(data) + 1.0 * (np.std(data))
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"
    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"
    plt.figure(figsize=(9, 9))
    plt.imshow(data, **kwargs)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.colorbar(label="Counts")
    plt.show()

# End
