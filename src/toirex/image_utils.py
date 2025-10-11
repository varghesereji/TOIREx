from photutils.centroids import centroid_2dg


def select_source(data, error=None, mask=None):
    centroid = centroid_2dg(data)
    # print("centroid", centroid)
    return centroid
