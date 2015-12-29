#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'build')

import os
import patches
import numpy as np
import cv2 as cv

sat = cv.imread('../../../data/mass_merged/test/sat/22828930_15.tiff')
map = cv.imread('../../../data/mass_merged/test/map/22828930_15.tif',
                cv.IMREAD_GRAYSCALE)[:, :, np.newaxis]

sat_p, map_p = patches.divide_to_patches(16, 92, 24, sat, map)
print(sat_p.shape, map_p.shape)

out_dir = 'test'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for i in range(sat_p.shape[0]):
    sp = sat_p[i]
    mp = map_p[i][:, :, 0]
    mp = np.array([mp == 0, mp == 1, mp == 2], dtype=np.uint8) * 255
    mp = mp.transpose((1, 2, 0))
    sp2 = sp.copy()
    sp2[92 / 2 - 24 / 2:92 / 2 + 24 / 2, 92 / 2 - 24 / 2:92 / 2 + 24 / 2] = mp
    img = np.hstack((sp, sp2))
    print(img.shape)

    cv.imwrite('{}/{}.png'.format(out_dir, i), img)
