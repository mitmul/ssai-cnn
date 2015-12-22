#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import imp
import argparse
import numpy as np
import cv2 as cv
from chainer import serializers, Variable, cuda


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
    xp = cuda.cupy if args.gpu >= 0 else np

    n_patch = middle.shape[0]
    n_side = int(np.ceil(np.sqrt(n_patch)))
    h, w = middle.shape[1:]
    canvas = xp.ones((n_side * h + (n_side + 1) * pad,
                      n_side * w + (n_side + 1) * pad),
                     dtype=np.uint8) * 125
    middle -= middle.min()
    middle /= middle.max()
    middle = (middle * 255).astype(xp.uint8)
    for i in range(n_patch):
        patch = middle[i]
        # patch -= patch.min()
        # patch /= patch.max()
        # patch = (patch * 255).astype(np.uint8)
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

    o_side = args.ortho_side
    l_side = args.label_side

    ortho = cv.imread('data/mass_merged/test/sat/24179065_15.tiff')
    h_limit, w_limit, _ = ortho.shape
    for y in range(0, h_limit // 4, l_side):
        for x in range(0, w_limit // 4, l_side):
            if (((y + o_side) > h_limit) or
                    ((x + o_side) > w_limit)):
                break
            out_fn = '{}/{}_{}'.format(args.out_dir, y, x)

            # ortho patch
            o_patch = ortho[y:y + o_side, x:x + o_side, :].astype(
                np.float32, copy=False)
            o_patch -= o_patch.reshape(-1, 3).mean(axis=0)
            o_patch /= o_patch.reshape(-1, 3).std(axis=0) + 1e-5
            o_patch = o_patch.transpose((2, 0, 1))[np.newaxis, :, :, :]

            b = Variable(xp.asarray(o_patch, dtype=xp.float32), volatile=True)
            middles = model.middle_layers(b)
            for name, middle in middles:
                middle = middle.data[0]
                if name == 'pred':
                    middle = middle.transpose((1, 2, 0))
                    tiled = middle * 255
                    tiled = tiled if not args.gpu >= 0 else xp.asnumpy(tiled)
                else:
                    tiled = tile_middle(middle)
                cv.imwrite('{}_{}.png'.format(out_fn, name), tiled)
