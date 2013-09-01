#!/usr/bin/env python

# Author: Igor Topcin <topcin@ime.usp.br>
# PTC5892 Processamento de Imagens Medicas
# POLI - University of Sao Paulo

# Implementation of the Kuan filter for multiplicative Gaussian noise, based
# on the 1985 paper.

# References:
# [1] D. T. Kuan, A. A. Sawchuk, T. C. Strand and P. Chavel, Adaptive Noise
# Smoothing Filter for Images with Signal-Dependent Noise. IEEE Transactions
# on Pattern Analysis and Machine Intelligence, Vol. 7, No. 2, 1985
# [2] Source code from PyRadar Project. https://github.com/PyRadar/pyradar,
# published under LGPL version 3.

import numpy as np
import scipy.stats as stats

DEFAULT_WINDOW_SIZE = 7
DEFAULT_CU = 0.25

def variation(window):
    """
    Calculates the coefficient of variation, that is, the ratio of the standard
    deviation to the mean.
    """
    return stats.variation(window, None)

def weight(window, cu):
    """
    Calculates the weight parameter of the Kuan filter. The weight defines
    weather the Kuan filter will act as an identity filter or as a mean filter.
    """
    ci = variation(window)
    ci_2 = ci * ci
    cu_2 = cu * cu

    if cu_2 > ci_2:
        w = 0.0 # do not allow negative values
    else:
        w = (1.0 - (cu_2 / ci_2)) / (1.0 + cu_2)

    return w

def filter(img, window_size = DEFAULT_WINDOW_SIZE, cu = DEFAULT_CU):
    """
    Filter the given image, using the passed in window size and coefficient of
    variation of a smooth area.
    Input arguments:
        img: a numpy array representing the image to be filtered.
        window_size: the size.
        cu: the coefficient of variation of a smooth region of img.
    Output:
        the filtered image.
    """

    img = np.float64(img)
    img_filtered = np.zeros_like(img)

    N, M = img.shape
    win_offset = window_size / 2

    for i in xrange(0, N):
        xleft = i - win_offset
        xright = i + win_offset

        # border conditions in x axis
        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N

        for j in xrange(0, M):
            yup = j - win_offset
            ydown = j + win_offset

            # border conditions in y axis
            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M

            # Calculate new value for pixel [i,j]
            window = img[xleft:xright, yup:ydown]
            w = weight(window, cu)
            img_filtered[i, j] = round((img[i, j] * w) + (window.mean() * (1.0 - w)))

    return img_filtered
