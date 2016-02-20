#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda
from chainer import serializers
from skimage import io

import VGG


def deprocess_image(x):
    if isinstance(x, cuda.cupy.ndarray):
        x = cuda.cupy.asnumpy(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--target_layer', type=str, default='conv3')
    parser.add_argument('--filter_num', type=int, default=9)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    model = VGG.VGG()
    serializers.load_hdf5('VGG.model', model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = cuda.cupy if args.gpu >= 0 else np

    img_width = args.img_size
    img_height = args.img_size
    target_layer = args.target_layer
    filter_num = args.filter_num
    pad = 5
    side = int(np.ceil(np.sqrt(filter_num)))
    canvas = np.zeros(((side + 1) * pad + img_height * side,
                       (side + 1) * pad + img_width * side, 3), dtype=np.uint8)

    for filter_index in range(filter_num):
        img = np.random.random((1, 3, img_height, img_width)) * 0.08 - 0.5
        img_link = chainer.Variable(xp.array(img, dtype=np.float32))

        for i in range(12):
            layer_output = model(img_link, target_layer)
            if filter_index == 0:
                layer_output = F.split_axis(
                    layer_output, [filter_index + 1], axis=1)[0]
            elif filter_index < filter_num - 1:
                layer_output = F.split_axis(
                    layer_output, [filter_index, filter_index + 1], axis=1)[1]
            else:
                layer_output = F.split_axis(
                    layer_output, [filter_index], axis=1)[1]

            img_link.zerograd()
            loss = F.sum(layer_output) / np.prod(layer_output.data.shape[2:])
            loss.backward()
            img_link.data += img_link.grad
        print(loss.data)

        y = deprocess_image(img_link.data[0])
        x_pos = filter_index % side
        y_pos = filter_index // side
        canvas[(y_pos + 1) * pad + y_pos * img_height:
               (y_pos + 1) * pad + y_pos * img_height + img_height,
               (x_pos + 1) * pad + x_pos * img_width:
               (x_pos + 1) * pad + x_pos * img_width + img_width, :] = y

    io.imsave('{}.png'.format(args.target_layer), canvas)
