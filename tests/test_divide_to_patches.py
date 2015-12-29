#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from utils.patches import divide_to_patches
import time
import numpy as np
import cv2 as cv

sat_im = cv.imread('data/mass_buildings/test/sat/22828930_15.tiff',
                   cv.IMREAD_COLOR)
map_im = cv.imread('data/mass_buildings/test/map/22828930_15.tif',
                   cv.IMREAD_GRAYSCALE)[:, :, np.newaxis]
print(sat_im.shape, map_im.shape)

stride = 16
sat_size = 92
map_size = 24

st = time.time()
sat_patches, map_patches = divide_to_patches(
    stride, sat_size, map_size, sat_im, map_im)
print(time.time() - st)

st = time.time()
sat_patches, map_patches = divide_to_patches(
    stride, sat_size, map_size, sat_im, map_im)
print(time.time() - st)

out_dir = 'data/test_patches'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for i in range(sat_patches.shape[0]):
    sp = sat_patches[i]
    mp = map_patches[i][:, :, 0]
    mp = np.array([mp == 0, mp == 1, mp == 2], dtype=np.uint8) * 255
    mp = mp.transpose((1, 2, 0))
    sp2 = sp.copy()
    sp2[92 / 2 - 24 / 2:92 / 2 + 24 / 2, 92 / 2 - 24 / 2:92 / 2 + 24 / 2] = mp
    img = np.hstack((sp, sp2))

    cv.imwrite('{}/{}.png'.format(out_dir, i), img)
print(i)
