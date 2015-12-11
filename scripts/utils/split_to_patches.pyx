#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython

UINT8 = np.uint8
ctypedef np.uint8_t UINT8_t


def divide_to_patches(int stride, int map_ch, int sat_size, int map_size,
                      np.ndarray[UINT8_t, ndim=3] sat_im,
                      np.ndarray[UINT8_t, ndim=3] map_im):
    cdef int height = sat_im.shape[0]
    cdef int width = sat_im.shape[1]
    cdef unsigned char[:, :, :] sat_patch
    cdef unsigned char[:, :, :] map_patch
    cdef int sum_patch_values = 0
    cdef int sum_pixel_values = 0

    cdef int n_patches = ((height // stride + 2) * ((width // stride + 2)))
    cdef unsigned char[:, :, :, :] map_patches
    map_patches = np.zeros((n_patches, map_size, map_size, 1), dtype=UINT8)
    cdef unsigned char[:, :, :, :] sat_patches
    sat_patches = np.zeros((n_patches, sat_size, sat_size, 3), dtype=UINT8)

    cdef int i = 0
    for y in range(0, height + stride, stride):
        for x in range(0, width + stride, stride):
            if (y + sat_size) > height:
                y = height - sat_size
            if (x + sat_size) > width:
                x = width - sat_size

            sat_patch = sat_im[y:y + sat_size, x:x + sat_size]
            map_patch = map_im[y + sat_size / 2 - map_size / 2:
                               y + sat_size / 2 + map_size / 2,
                               x + sat_size / 2 - map_size / 2:
                               x + sat_size / 2 + map_size / 2]

            # exclude patch including big white region
            sum_patch_values = 0
            for yy in range(sat_size):
                for xx in range(sat_size):
                    sum_pixel_values = 0
                    for cc in range(3):
                        sum_pixel_values += sat_patch[yy, xx, cc]
                    if sum_pixel_values == 255 * 3:
                        sum_patch_values += 1
            if sum_patch_values > 16:
                continue

            sat_patches[i, :, :, :] = sat_patch
            map_patches[i, :, :, :] = map_patch
            i += 1

    return sat_patches, map_patches, i
