#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython


def relax_precision(np.ndarray[np.uint8_t, ndim=2] predict,
                    np.ndarray[np.uint8_t, ndim=2] label,
                    int relax):
    cdef int h_lim = predict.shape[0]
    cdef int w_lim = predict.shape[1]
    cdef int st_y
    cdef int en_y
    cdef int st_x
    cdef int en_x
    cdef int predict_val
    cdef int label_val
    cdef int flag
    cdef int true_positive = 0
    for y in range(h_lim):
        for x in range(w_lim):
            predict_val = predict[y, x]
            if predict_val == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1
                flag = 0
                for yy in range(st_y, en_y):
                    for xx in range(st_x, en_x):
                        label_val = label[yy, xx]
                        if label_val == 1:
                            true_positive += 1
                            flag = 1
                            break
                    if flag == 1:
                        break

    return true_positive


def relax_recall(np.ndarray[np.uint8_t, ndim=2] predict,
                 np.ndarray[np.uint8_t, ndim=2] label,
                 int relax):
    cdef int h_lim = predict.shape[0]
    cdef int w_lim = predict.shape[1]
    cdef int st_y
    cdef int en_y
    cdef int st_x
    cdef int en_x
    cdef int true_positive = 0
    cdef int predict_val
    cdef int label_val
    cdef int flag
    for y in range(h_lim):
        for x in range(w_lim):
            label_val = label[y, x]
            if label_val == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1
                flag = 0
                for yy in range(st_y, en_y):
                    for xx in range(st_x, en_x):
                        predict_val = predict[yy, xx]
                        if predict_val == 1:
                            true_positive += 1
                            flag = 1
                            break
                    if flag == 1:
                        break

    return true_positive
