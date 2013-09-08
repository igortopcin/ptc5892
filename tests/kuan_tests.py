#!/usr/bin/env python

# Author: Igor Topcin <topcin@ime.usp.br>
# PTC5892 Processamento de Imagens Medicas
# POLI - University of Sao Paulo

# This is the test package for this project
# In order do run tests, go into tests directory and run "nosetests".

import numpy as np
import Image
import ptc5892.kuan as kuan
import matplotlib.pyplot as plt

from nose.tools import *
from ptc5892.canny import canny
from ptc5892.fom import fom
from scipy.ndimage import distance_transform_edt

def filter(im_orig_array, im_gold_array, window_size, cu, test_index = 1):
	"""
	This function encapsulates all the calls that are necessary in order to
	test kuan filter in a noisy image.
	"""
	# plot original image
	plt.figure(test_index * 10 + 1)
	plt.subplot(131)
	plt.imshow(im_orig_array, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('noisy')

	# apply kuan filter using the computed coefficient of variation of a smooth
	# region in the original noisy image
	im_filtered_array = kuan.filter(im_orig_array, window_size, cu)

	plt.figure(test_index * 10 + 1)
	plt.subplot(132)
	plt.imshow(im_filtered_array, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('filtered, Cu=%s' % cu)

	# plot the gold standard image
	plt.figure(test_index * 10 + 1)
	plt.subplot(133)
	plt.imshow(im_gold_array, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('gold standard')

	# detect the edges of the filtered image
	im_filtered_edges = canny(im_filtered_array, 0.1, 20, 50)

	plt.figure(test_index * 10 + 2)
	plt.subplot(131)
	plt.imshow(im_filtered_edges, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('filtered edges')

	# detect the edges of the gold standard using the same edge detector
	im_gold_edges = canny(im_gold_array, 0.1, 20, 50)

	plt.figure(test_index * 10 + 2)
	plt.subplot(132)
	plt.imshow(im_gold_edges, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('gold std edges')

	# Plot the distance transform for the edges in the gold standard image
	im_gold_dist = distance_transform_edt(np.invert(im_gold_edges))
	f = fom(im_filtered_array, im_gold_array)
	plt.figure(test_index * 10 + 2)
	plt.subplot(133)
	plt.imshow(im_gold_dist)
	plt.axis('off')
	plt.title('dist transf gold,fom=%s' % f)

	plt.show()

def test_image1():
	# read the image with speckle
	im_orig = Image.open('imgs/fig_geom_speckled.tif').convert('L')
	im_orig_array = np.array(im_orig)

	# calculate the coefficient of variation of a smooth region of the image.
	cu = kuan.variation(im_orig_array[50:100,300:350])

	# read the gold standard image
	im_gold = Image.open('imgs/fig_geom_goldStd.tif').convert('L')
	im_gold_array = np.array(im_gold)
	filter(im_orig_array, im_gold_array, 11, cu, 1)

def test_image2():
	# read the original image, with Gaussian noise
	im_orig = Image.open('imgs/cistosRuidoGaussian10dB.jpg').convert('L')
	im_orig_array = np.array(im_orig)

	# calculate the coefficient of variation of a smooth region of the image.
	cu = kuan.variation(im_orig_array[50:100,50:100])

	# read the gold standard image
	im_gold = Image.open('imgs/cistosGoldStd.tif').convert('L')
	im_gold_array = np.array(im_gold)

	filter(im_orig_array, im_gold_array, 9, cu, 2)

def test_image3():
	# read the original image, with Gaussian noise
	im_orig = Image.open('imgs/cistosSpeckle.tif').convert('L')
	im_orig_array = np.array(im_orig)

	# calculate the coefficient of variation of a smooth region of the image.
	cu = kuan.variation(im_orig_array[50:100,50:100])

	# read the gold standard image
	im_gold = Image.open('imgs/cistosGoldStd.tif').convert('L')
	im_gold_array = np.array(im_gold)

	filter(im_orig_array, im_gold_array, 13, cu, 3)
