#!/usr/bin/env python

import os
import imp
import argparse
import chainer
import cv2 as cv
import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import serializers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--layer', type=str, default='conv1')
    parser.add_argument('--img_fn', type=str,
                        default='data/mass_merged/trans_test/107.jpg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--in_size', type=int, default=64)
    parser.add_argument('--x0_sigma', type=str, default='data/x0_sigma.txt')
    args = parser.parse_args()

    for line in open(args.x0_sigma):
        args.x0_sigma = float(line.strip())
        break

    return args


def preprocess(img):
    if img.shape[2] != 3:
        raise ValueError('Wrong number of channels.')
    img = img.astype(np.float)
    mean = img.reshape((-1, 3)).mean(axis=0)
    std = img.reshape((-1, 3)).std(axis=0)
    img -= mean
    img /= std
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)

    return img, mean, std


def deprocess(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


class InvertLoss(object):

    def __init__(self, model, in_size, layer, x0_sigma, img, lambda_tv=5,
                 lambda_lp=4e-10, beta=2, p=6):
        xp = cuda.cupy if args.gpu >= 0 else np
        self.model = model
        Wh_data = xp.array([[[[1], [-1]]]], dtype='f')
        Ww_data = xp.array([[[[1, -1]]]], dtype='f')
        self.Wh = chainer.Variable(Wh_data)
        self.Ww = chainer.Variable(Ww_data)
        self.in_size = in_size
        self.layer = layer
        self.x0_sigma = x0_sigma
        self.lambda_tv = lambda_tv
        self.lambda_lp = lambda_lp
        self.beta = beta
        self.p = p

        self.create_target(img)

    def create_target(self, img):
        self.x0_data = xp.asarray(img[np.newaxis].copy())
        x0 = chainer.Variable(self.x0_data, volatile=True)

        # Extract feature from target image
        self.y0 = self.extract_feature(x0)
        self.y0.volatile = False
        y0_sigma = np.linalg.norm(cuda.to_cpu(self.y0.data))
        self.lambda_euc = self.x0_sigma ** 2 / y0_sigma ** 2

    def extract_feature(self, x):
        middles = dict(self.model.middle_layers(x))

        return middles[self.layer]

    def tvh(self, x):
        return F.convolution_2d(x, W=self.Wh)

    def tvw(self, x):
        return F.convolution_2d(x, W=self.Ww)

    def tv_norm(self, x):
        diffh = self.tvh(F.reshape(x, (3, 1, self.in_size, self.in_size)))
        diffw = self.tvw(F.reshape(x, (3, 1, self.in_size, self.in_size)))
        tv = (F.sum(diffh ** 2) + F.sum(diffw ** 2)) ** (self.beta / 2.)

        return tv

    def __call__(self, x):
        y = self.extract_feature(x)
        e = F.mean_squared_error(self.y0, y)
        tv = self.tv_norm(x)
        loss = (self.lambda_euc * float(self.y0.data.size) * e +
                self.lambda_tv * tv + self.lambda_lp * F.sum(x ** self.p))
        print(loss.data)

        return loss


if __name__ == '__main__':
    args = get_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np

    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model
    model.train = False
    serializers.load_hdf5(args.param, model)
    if args.gpu >= 0:
        model.to_gpu()

    # Initialize parameters
    img = cv.imread(args.img_fn)
    img = img[:, :img.shape[1] / 2, :]
    cv.imwrite('input.png', img)
    img, mean, std = preprocess(img)

    loss = InvertLoss(model, args.in_size, args.layer, args.x0_sigma, img)

    # Image plane to be optimized
    x_data = np.random.randn(*loss.x0_data.shape).astype('f')
    x_data = x_data / np.linalg.norm(x_data) * args.x0_sigma
    x_data = xp.asarray(x_data)
    x_link = chainer.links.Parameter(x_data)

    # Define hyper-parameters
    learning_rate = 0.0015 * np.array(
        [0.1] * 10000 +
        [0.05] * 5000 +
        [0.01] * 5000, dtype='f')
    momentum = 0.9

    opt = optimizers.MomentumSGD(momentum=momentum)
    opt.setup(x_link)

    lr_prev = 0
    for lr in learning_rate:
        if lr != lr_prev:
            opt.lr = lr
            lr_prev = lr
            opt.momentum = 0
            print('lr:', opt.lr)
        else:
            opt.momentum = 0.9

        x = x_link.W
        opt.update(loss, x)

    result = deprocess(cuda.to_cpu(x.data)[0], mean, std)
    cv.imwrite('{}_result.png'.format(args.layer), result)
