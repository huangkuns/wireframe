import numpy as np
import pdb


DEBUG=False

def generate_anchors(grid_size, stride=8.):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    grid_h, grid_w = grid_size
    yy, xx = np.meshgrid(stride * np.arange(grid_h), stride * np.arange(grid_w))
    anchors = np.stack((yy, xx), axis=2)
    #print anchors.shape

    anchors = anchors + stride/2.
    if DEBUG:
        print(anchors)
        print(anchors.reshape((-1, 2)))
    return anchors

def generate_bin_anchors(num_bin):
    assert int(360./num_bin) * num_bin == 360
    bin_width = 360./float(num_bin)
    bin_anchors = [x * bin_width for x in range(int(360/bin_width))]
    
    if DEBUG:
        print(bin_anchors)
    return bin_anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    generate_anchors((3,3), stride=2)
    generate_bin_anchors(12)
    exit()
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    #from IPython import embed; embed()

