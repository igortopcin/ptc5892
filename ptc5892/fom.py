#!/usr/bin/env python

# Author: Igor Topcin <topcin@ime.usp.br>
# PTC5892 Processamento de Imagens Medicas
# POLI - University of Sao Paulo

# Implementation of the 

# References:
# [1] W. K. Pratt, Digital Image Processing. New York: Wiley, 1977
# [2] Y. Yu and S. T. Acton, Speckle Reducing Anisotropic Diffusion.
# IEEE Transactions on Image Processing, Vol. 11, No. 11, 2002

import numpy as np

from canny import canny
from scipy.ndimage import distance_transform_edt

DEFAULT_ALPHA = 1.0 / 9

def fom(img, img_gold_std, alpha = DEFAULT_ALPHA):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    edges_img = canny(img, 0.1, 20, 50)
    edges_gold = canny(img_gold_std, 0.1, 20, 50)
    
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(edges_gold))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    N, M = img.shape

    for i in xrange(0, N):
        for j in xrange(0, M):
            if edges_img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))    

    return fom
