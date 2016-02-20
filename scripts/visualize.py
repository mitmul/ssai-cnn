#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import imp
import os

import matplotlib.pyplot as plt
import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers

import cv2 as cv
from train import get_cursor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--ortho_original_side', type=int, default=92)
    parser.add_argument('--label_original_side', type=int, default=24)
    parser.add_argument('--ortho_side', type=int, default=64)
    parser.add_argument('--label_side', type=int, default=16)
    args = parser.parse_args()

    return args


def tile_W(W, pad=1):
    xp = cuda.cupy if args.gpu >= 0 else np

    n_patch = W.shape[0]
    n_side = int(np.ceil(np.sqrt(n_patch)))
    h, w = W.shape[2:]
    canvas = np.zeros((n_side * h + (n_side + 1) * pad,
                       n_side * w + (n_side + 1) * pad, 3),
                      dtype=np.uint8)
    W = W if not args.gpu >= 0 else xp.asnumpy(W)
    # W -= W.min()
    # W /= W.max()
    # W = (W * 255).astype(np.uint8)
    for i in range(n_patch):
        patch = W[i].transpose(1, 2, 0)
        patch -= patch.min()
        patch /= patch.max()
        patch = (patch * 255).astype(np.uint8)

        y = i // n_side
        y = pad * (y + 1) + y * h
        x = i % n_side
        x = pad * (x + 1) + x * w

        canvas[y:y + h, x:x + w] = patch

    return canvas


def tile_middle(name, middle, pad=1):
    xp = cuda.cupy if args.gpu >= 0 else np

    n_patch = middle.shape[0]
    n_side = int(np.ceil(np.sqrt(n_patch)))
    if middle.ndim == 3:
        h, w = middle.shape[1:]
    else:
        h, w, = 1, 1
    canvas = xp.ones((n_side * h + (n_side + 1) * pad,
                      n_side * w + (n_side + 1) * pad),
                     dtype=np.uint8) * 125
    if name != 'reshape' and name != 'cis':
        middle -= middle.min()
        middle /= middle.max()
        middle = (middle * 255).astype(xp.uint8)
    for i in range(n_patch):
        patch = middle[i]
        if name == 'reshape':
            patch -= patch.min()
            patch /= patch.max()
            patch = (patch * 255).astype(np.uint8)
        if name == 'cis' and i != 0:
            patch -= patch.min()
            patch /= patch.max()
            patch = (patch * 255).astype(np.uint8)

        y = i // n_side
        y = pad * (y + 1) + y * h
        x = i % n_side
        x = pad * (x + 1) + x * w
        canvas[y:y + h, x:x + w] = patch

    canvas = canvas if not args.gpu >= 0 else xp.asnumpy(canvas)

    return canvas


if __name__ == '__main__':
    args = get_args()
    module = os.path.splitext(os.path.basename(args.model))[0]
    model = imp.load_source(module, args.model).model
    serializers.load_hdf5(args.param, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    model.train = False

    xp = cuda.cupy if args.gpu >= 0 else np

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
            o_patch -= o_patch.reshape((-1, 3)).mean(axis=0)
            o_patch /= o_patch.reshape((-1, 3)).std(axis=0)
            o_patch = np.expand_dims(o_patch.transpose((2, 0, 1)), axis=0)
            o_patch = xp.asarray(o_patch, dtype=xp.float32)
            middles = model.middle_layers(Variable(o_patch, volatile=True))

            for name, middle in middles:
                print(name)
                middle = middle.data[0]
                if name == 'pred':
                    middle = middle.transpose((1, 2, 0))
                    tiled = middle * 255
                    tiled = tiled if not args.gpu >= 0 else xp.asnumpy(tiled)
                # elif name == 'reshape':
                #     plt.clf()
                #     a = middle[0] if not args.gpu >= 0 \
                #         else xp.asnumpy(middle[0])
                #     a = a.ravel()
                #     print(middle[0].std(), middle[1].std(), middle[2].std())
                #     plt.hist(a)
                #     print('hist')
                #     plt.savefig('{}_0_hist.png'.format(out_fn))
                else:
                    tiled = tile_middle(name, middle)
                cv.imwrite('{}_{}.png'.format(out_fn, name), tiled)

            i += 1
            if i > args.num:
                break

        o_ret = o_cur.next()
        l_ret = l_cur.next()
        if ((not o_ret) and (not l_ret)):
            break
