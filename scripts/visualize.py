#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import imp
import argparse
import numpy as np
import cv2 as cv
from chainer import serializers, Variable
from train import get_cursor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--ortho_original_side', type=int, default=92)
    parser.add_argument('--label_original_side', type=int, default=24)
    parser.add_argument('--ortho_side', type=int, default=64)
    parser.add_argument('--label_side', type=int, default=16)
    args = parser.parse_args()

    return args


def tile_W(W, pad=1):
    n_patch = W.shape[0]
    n_side = int(np.ceil(np.sqrt(n_patch)))
    h, w = W.shape[2:]
    canvas = np.zeros((n_side * h + (n_side + 1) * pad,
                       n_side * w + (n_side + 1) * pad, 3),
                      dtype=np.uint8)
    W -= W.min()
    W /= W.max()
    W = (W * 255).astype(np.uint8)
    for i in range(n_patch):
        patch = W[i].transpose(1, 2, 0)
        y = i // n_side
        y = pad * (y + 1) + y * h
        x = i % n_side
        x = pad * (x + 1) + x * w

        canvas[y:y + h, x:x + w] = patch

    return canvas


def tile_middle(middle, pad=1):
    n_patch = middle.shape[0]
    n_side = int(np.ceil(np.sqrt(n_patch)))
    h, w = middle.shape[1:]
    canvas = np.ones((n_side * h + (n_side + 1) * pad,
                      n_side * w + (n_side + 1) * pad),
                     dtype=np.uint8) * 125
    for i in range(n_patch):
        patch = middle[i]
        patch -= patch.min()
        patch /= patch.max()
        patch = (patch * 255).astype(np.uint8)
        y = i // n_side
        y = pad * (y + 1) + y * h
        x = i % n_side
        x = pad * (x + 1) + x * w

        canvas[y:y + h, x:x + w] = patch

    return canvas


if __name__ == '__main__':
    args = get_args()
    module = os.path.splitext(os.path.basename(args.model))[0]
    model = imp.load_source(module, args.model).model
    serializers.load_hdf5(args.param, model)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    tile_conv1 = tile_W(model.conv1.W.data)
    cv.imwrite('{}/conv1_W.png'.format(args.out_dir), tile_conv1)

    o_side = args.ortho_original_side
    l_side = args.label_original_side
    om_side = args.ortho_side
    lm_side = args.label_side

    o_cur, o_txn, _ = get_cursor('data/mass_merged/lmdb/test_sat')
    l_cur, l_txn, _ = get_cursor('data/mass_merged/lmdb/test_map')

    i = 0
    while True:
        o_key, o_val = o_cur.item()
        l_key, l_val = l_cur.item()
        if o_key != l_key:
            raise ValueError(
                'Keys of ortho and label patches are different: '
                '{} != {}'.format(o_key, l_key))

        # prepare patch
        o_patch = np.fromstring(
            o_val, dtype=np.uint8).reshape((o_side, o_side, 3))
        l_patch = np.fromstring(
            l_val, dtype=np.uint8).reshape((l_side, l_side, 1))

        num_b, num_r = np.sum(l_patch == 1), np.sum(l_patch == 2)
        if num_b > 0 and num_r > 0:
            out_fn = '{}/{}'.format(args.out_dir, o_key.decode('utf-8'))

            # crop center
            o_patch = o_patch[o_side / 2 - om_side / 2:
                              o_side / 2 + om_side / 2,
                              o_side / 2 - om_side / 2:
                              o_side / 2 + om_side / 2]
            l_patch = l_patch[l_side / 2 - lm_side / 2:
                              l_side / 2 + lm_side / 2,
                              l_side / 2 - lm_side / 2:
                              l_side / 2 + lm_side / 2]
            l_patch = l_patch.reshape((lm_side ** 2,))
            l_patch = np.hstack([l_patch == 0, l_patch == 1, l_patch == 2])
            l_patch = l_patch.reshape((3, lm_side, lm_side))
            l_patch = l_patch.transpose((1, 2, 0)).astype(np.uint8) * 255
            cv.imwrite('{}_ortho.png'.format(out_fn), o_patch)
            cv.imwrite('{}_label.png'.format(out_fn), l_patch)

            o_patch = o_patch.astype(np.float32)
            o_patch -= o_patch.reshape((3, -1)).mean(axis=1)
            o_patch /= o_patch.reshape((3, -1)).std(axis=1)
            o_patch = np.expand_dims(o_patch.transpose((2, 0, 1)), axis=0)
            middles = model.middle_layers(Variable(o_patch))

            for name, middle in middles:
                middle = middle.data[0]
                tiled = tile_middle(middle)
                cv.imwrite('{}_{}.png'.format(out_fn, name), tiled)

            i += 1
            if i > args.num:
                break

        o_ret = o_cur.next()
        l_ret = l_cur.next()
        if ((not o_ret) and (not l_ret)):
            break
