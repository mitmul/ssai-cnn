#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import lmdb
import argparse
import numpy as np
import cv2 as cv
from utils.transformer import transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument('--ortho_db', type=str)
    parser.add_argument('--label_db', type=str)
    parser.add_argument('--out_dir', type=str)

    # Dataset info
    parser.add_argument('--ortho_original_side', type=int, default=92)
    parser.add_argument('--label_original_side', type=int, default=24)
    parser.add_argument('--ortho_side', type=int, default=64)
    parser.add_argument('--label_side', type=int, default=16)

    # Options for data augmentation
    parser.add_argument('--fliplr', type=int, default=1)
    parser.add_argument('--rotate', type=int, default=1)
    parser.add_argument('--angle', type=int, default=90)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--crop', type=int, default=1)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    ortho_env = lmdb.open(args.ortho_db)
    ortho_txn = ortho_env.begin(write=False, buffers=False)
    ortho_cur = ortho_txn.cursor()
    ortho_cur.next()

    label_env = lmdb.open(args.label_db)
    label_txn = label_env.begin(write=False, buffers=False)
    label_cur = label_txn.cursor()
    label_cur.next()

    o_batch = []
    l_batch = []
    for i in range(128):
        o_key, o_val = ortho_cur.item()
        l_key, l_val = label_cur.item()

        o_patch = np.fromstring(o_val, dtype=np.uint8).reshape(
            (args.ortho_original_side, args.ortho_original_side, 3))
        l_patch = np.fromstring(l_val, dtype=np.uint8).reshape(
            (args.label_original_side, args.label_original_side, 1))

        o_batch.append(o_patch)
        l_batch.append(l_patch)

        ortho_cur.next()
        label_cur.next()

    st = time.time()
    o_batch = np.asarray(o_batch, dtype=np.uint8)
    l_batch = np.asarray(l_batch, dtype=np.uint8)

    o_aug, l_aug = transform(
        o_batch, l_batch, args.fliplr, args.rotate, args.norm, args.ortho_side,
        args.ortho_side, 3, args.label_side, args.label_side)

    print(time.time() - st, 'sec', o_aug.shape, l_aug.shape)

    for i, (o, l) in enumerate(zip(o_aug, l_aug)):
        o = o.transpose((1, 2, 0))
        o = o - o.min()
        o = o / o.max()
        o *= 255

        l = l.reshape(-1)
        l = np.hstack([l == 0, l == 1, l == 2])
        l = l.reshape(
            (3, 16, 16)).transpose((1, 2, 0)).astype(np.uint8) * 255

        canvas = np.zeros((args.ortho_side, args.ortho_side * 2, 3))
        canvas[:, :args.ortho_side, :] = o
        canvas[:, args.ortho_side:, :] = o
        canvas[args.ortho_side / 2 - args.label_side / 2:
               args.ortho_side / 2 + args.label_side / 2,
               args.ortho_side + args.ortho_side / 2 - args.label_side / 2:
               args.ortho_side + args.ortho_side / 2 + args.label_side / 2,
               :] = l
        canvas = canvas.astype(np.uint8)

        cv.imwrite('{}/{}.jpg'.format(args.out_dir, i), canvas)
