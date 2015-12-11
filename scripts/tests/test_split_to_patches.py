#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '../utils')
import split_to_patches
import time
import numpy as np
import cv2 as cv

sat_im = cv.imread(
    '../../data/mass_buildings/test/sat/22828930_15.tiff', cv.IMREAD_COLOR)
map_im = cv.imread(
    '../../data/mass_buildings/test/map/22828930_15.tif', cv.IMREAD_GRAYSCALE)
map_im = map_im[:, :, np.newaxis]
print(sat_im.shape, map_im.shape)
stride = 16
map_ch = 1
sat_size = 92
map_size = 24

height, width, channels = sat_im.shape
print((height // stride + 2) * (width // stride + 2))

st = time.time()
sat_patches, map_patches, n_patches = split_to_patches.divide_to_patches(
    stride, map_ch, sat_size, map_size, sat_im, map_im)
print(time.time() - st)
sat_patches = np.asarray(sat_patches, dtype=np.uint8)
map_patches = np.asarray(map_patches, dtype=np.uint8)
print(sat_patches.shape, map_patches.shape, n_patches)
if not os.path.exists('test'):
    os.mkdir('test')
for i in range(len(sat_patches)):
    sat_patch = np.asarray(sat_patches[i], dtype=np.uint8)
    map_patch = np.asarray(map_patches[i], dtype=np.uint8)
    cv.imwrite('test/{}_sat_patch.png'.format(i), sat_patch)
    cv.imwrite('test/{}_map_patch.png'.format(i), map_patch * 255)

print('cython:{}'.format(time.time() - st))
print(i)
