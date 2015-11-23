#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import lmdb
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ortho_db', type=str)
    parser.add_argument('--label_db', type=str)
    parser.add_argument('--out_dir', type=str)
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

    for i in range(100):
        o_key, o_val = ortho_cur.item()
        l_key, l_val = label_cur.item()

        o_patch = np.fromstring(
            o_val, dtype=np.uint8).reshape((92, 92, 3))
        l_patch = np.fromstring(
            l_val, dtype=np.uint8).reshape((24 * 24))
        l_patch = np.hstack([l_patch == 0, l_patch == 1, l_patch == 2])
        l_patch = l_patch.reshape(
            (3, 24, 24)).transpose((1, 2, 0)).astype(np.uint8) * 255

        canvas = np.zeros((92, 92 * 2, 3))
        canvas[:, :92, :] = o_patch
        canvas[:, 92:, :] = o_patch
        canvas[92 / 2 - 24 / 2:92 / 2 + 24 / 2,
               92 + 92 / 2 - 24 / 2:92 + 92 / 2 + 24 / 2, :] = l_patch
        canvas = canvas.astype(np.uint8)

        cv.imwrite('{}/{}.jpg'.format(args.out_dir, o_key), canvas)

        ortho_cur.next()
        label_cur.next()
