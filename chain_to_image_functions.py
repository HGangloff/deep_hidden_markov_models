'''
1D sequence <-> image, using an hilbert curve
requires the sequence have length equal to a power of 2
based on hilbertcurve.py from the package https://github.com/galtay/hilbertcurve
'''
import numpy as np
from hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_hilbertcurve_path(image_border_length):
    '''
    Given image_border_length, the length of a border of a square image, we
    compute path of the hilbert peano curve going through this image

    Note that the border length must be a power of 2.

    Returns a list of the coordinates of the pixel that must be visited (in
    order !)
    '''
    path = []
    p = int(np.log2(image_border_length))
    hilbert_curve = HilbertCurve(p, 2)
    path = []
    print("Compute path for shape ({0},{1})".format(image_border_length,
        image_border_length))
    for i in range(image_border_length ** 2):
        coords = hilbert_curve.coordinates_from_distance(i)
        path.append([coords[0], coords[1]])

    return path

def chain_to_image(X_ch, masked_peano_img=None):
    '''
    X_ch is an unidimensional array (a chain !) whose length is 2^(2*N) with N non negative
    integer.
    We transform X_ch to a 2^N * 2^N image following the hilbert peano curve
    '''
    if masked_peano_img is None:
        image_border_length = int(np.sqrt(X_ch.shape[0]))
        path = get_hilbertcurve_path(image_border_length)
        masked_peano_img = np.zeros((image_border_length, image_border_length))
    else:
        image_border_length = masked_peano_img.shape[0]
        path = get_hilbertcurve_path(image_border_length)
        
    X_img = np.empty((image_border_length, image_border_length))
    offset = 0
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_img[coords[0], coords[1]] = X_ch[idx - offset]
        else:
            offset += 1
            X_img[coords[0], coords[1]] = -1

    return X_img

def image_to_chain(X_img, masked_peano_img=None):
    '''
    X_img is a 2^N * 2^N image with N non negative integer.
    We transform X_img to a 2^(2*N) unidimensional vector (a chain !)
    following the hilbert peano curve
    '''
    path = get_hilbertcurve_path(X_img.shape[0])

    if masked_peano_img is None:
        masked_peano_img = np.zeros((X_img.shape[0], X_img.shape[1]))

    X_ch = []
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_ch.append(X_img[coords[0], coords[1]])

    return np.array(X_ch)

