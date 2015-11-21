#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import lmdb
import argparse
import numpy as np
from skimage import io
from transform import Transform
from create_args import create_args

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
    parser.add_argument('--channels', type=int, default=1)

    # Options for data augmentation
    parser.add_argument('--fliplr', type=int, default=1)
    parser.add_argument('--rotate', type=int, default=1)
    parser.add_argument('--angle', type=int, default=90)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--clip', type=int, default=1)

    args = create_args()
    trans = Transform(fliplr=bool(args.fliplr),
                      rotate=bool(args.rotate),
                      angle=args.angle,
                      norm=bool(args.norm),
                      clip=bool(args.clip),
                      ortho_side=args.ortho_side,
                      label_side=args.label_side)

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

        o_patch = np.fromstring(o_val, dtype=np.uint8).reshape((3, 96, 96))
        l_patch = np.fromstring(l_val, dtype=np.uint8).reshape(
            (args.channels, 24, 24))
        o_patch = o_patch.transpose((1, 2, 0))
        l_patch = l_patch.transpose((1, 2, 0))
        o_aug, l_aug = trans.transform(o_patch, l_patch)
        o_aug -= o_aug.min()
        o_aug /= o_aug.max()
        o_aug *= 255

        canvas = np.zeros((args.ortho_side, args.ortho_side * 2, 3))
        canvas[:, :args.ortho_side, :] = o_aug
        canvas[:, args.ortho_side:, :] = o_aug
        canvas[args.ortho_side / 2 - args.label_side / 2:
               args.ortho_side / 2 + args.label_side / 2,
               args.ortho_side + args.ortho_side / 2 - args.label_side / 2:
               args.ortho_side + args.ortho_side / 2 + args.label_side / 2,
               :] = l_aug * 255.0
        canvas = canvas.astype(np.uint8)

        io.imsave('{}/{}.jpg'.format(args.out_dir, o_key), canvas)

        ortho_cur.next()
        label_cur.next()
