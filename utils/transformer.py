#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from utils.transform import batch_transform


def transform(sat_batch, map_batch, fliplr, rotate, norm, sat_out_h, sat_out_w,
              sat_channels, map_out_h, map_out_w):
    if sat_batch.dtype != np.uint8:
        sat_batch = sat_batch.astype(np.uint8)
    if map_batch.dtype != np.uint8:
        map_batch = map_batch.astype(np.uint8)
    if sat_batch.ndim != 4:
        raise ValueError('sat_batch.ndim != 4')
    if map_batch.ndim != 4:
        raise ValueError('map_batch.ndim != 4')
    if map_batch.shape[3] != 1:
        raise ValueError('map_batch.shape[3] != 1')

    num = sat_batch.shape[0]
    sat_aug = np.zeros(
        (num, sat_out_h, sat_out_w, sat_channels), dtype=np.float32)
    map_aug = np.zeros((num, map_out_h, map_out_w, 1), dtype=np.float32)

    batch_transform(sat_batch, map_batch, sat_aug, map_aug, fliplr, rotate,
                    norm, sat_out_h, sat_out_w, sat_channels, map_out_h,
                    map_out_w)

    sat_aug = sat_aug.astype(np.float32).transpose((0, 3, 1, 2))
    map_aug = map_aug[:, :, :, 0].astype(np.int32)

    return (sat_aug, map_aug)

if __name__ == '__main__':
    imgs = []
    labels = []
    for i in range(128):
        img = np.ones((92, 92, 3), dtype=np.uint8) * 255
        imgs.append(img)

        label = np.ones((24, 24, 1), dtype=np.uint8)
        label[i * 10:i * 10 + 10, i * 10:i * 10 + 10, 1:] = 0
        labels.append(label)

    imgs = np.asarray(imgs, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)

    st = time.time()
    o_aug, l_aug = transform(imgs, labels, True, True, True,
                             64, 64, 3, 16, 16)
    print(time.time() - st)

    st = time.time()
    o_aug, l_aug = transform(imgs, labels, True, True, True,
                             64, 64, 3, 16, 16)
    print(time.time() - st)
