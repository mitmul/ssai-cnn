#!/usr/bin/env python

import argparse
import imp
import os
import re

import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import cv2 as cv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--layer', type=str, default='conv1')
    parser.add_argument('--img_fn', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--in_size', type=int, default=64)
    parser.add_argument('--x0_sigma', type=str, default='data/x0_sigma.txt')
    parser.add_argument('--lambda_tv', type=float, default=0.5)  # 0.5
    parser.add_argument('--lambda_lp', type=float, default=4e-10)  # 4e-10
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--p', type=float, default=6)
    parser.add_argument('--adam_alpha', type=float, default=0.1)
    parser.add_argument('--channels', type=int, default=-1)
    parser.add_argument('--max_iter', type=int, default=10000)
    args = parser.parse_args()

    for line in open(args.x0_sigma):
        args.x0_sigma = float(line.strip())
        break

    np.random.seed(args.seed)

    return args


class InvertFeature(object):

    def __init__(self, args):
        xp = cuda.cupy if args.gpu >= 0 else np
        xp.random.seed(args.seed)
        Wh_data = xp.array([[[[1], [-1]]]], dtype='f')
        Ww_data = xp.array([[[[1, -1]]]], dtype='f')
        self.Wh = chainer.Variable(Wh_data)
        self.Ww = chainer.Variable(Ww_data)
        self.args = args

        self.load_model()
        self.create_dir()
        self.get_img_var()
        self.create_target()
        self.create_image_plane()
        self.prepare_optimizer()
        self.create_lr_schedule()

    def load_model(self):
        model_fn = os.path.basename(self.args.model)
        self.model = imp.load_source(
            model_fn.split('.')[0], self.args.model).model
        self.model.train = False
        serializers.load_hdf5(self.args.param, self.model)
        if self.args.gpu >= 0:
            self.model.to_gpu()

    def create_dir(self):
        out_dir = os.path.dirname(self.args.param)
        epoch = int(re.search('epoch-([0-9]+)',
                              os.path.basename(self.args.param)).groups()[0])
        self.out_dir = '{}/inv-{}'.format(out_dir, epoch)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def get_img_var(self):
        self.img_id = int(
            re.search('([0-9]+).jpg', self.args.img_fn).groups()[0])

        img = cv.imread(self.args.img_fn)
        cv.imwrite('{}/{}_original.png'.format(self.out_dir, self.img_id), img)
        img = img[:, :img.shape[1] / 2, :]
        cv.imwrite('{}/{}_input.png'.format(self.out_dir, self.img_id), img)
        self.preprocess(img)

    def preprocess(self, img):
        if img.shape[2] != 3:
            raise ValueError('Wrong number of channels.')
        img = img.astype(np.float)
        self.mean = img.reshape((-1, 3)).mean(axis=0)
        self.std = img.reshape((-1, 3)).std(axis=0)
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        self.img = img.astype(np.float32)

    def deprocess(self, img):
        img = img.transpose(1, 2, 0)
        img *= self.std
        img += self.mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def extract_feature(self, x):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        middles = dict(self.model.middle_layers(x))
        middle = middles[self.args.layer]
        if self.args.channels < 0:
            return middle
        else:
            m = middle.data[:, self.args.channels, :, :]
            m = xp.expand_dims(m, axis=1)
            middle = chainer.Variable(m)
            return middle

    def create_target(self):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        self.x0_data = xp.asarray(self.img[np.newaxis].copy())
        x0 = chainer.Variable(self.x0_data, volatile=True)

        # Extract feature from target image
        self.y0 = self.extract_feature(x0)
        self.y0.volatile = False
        y0_sigma = np.linalg.norm(cuda.to_cpu(self.y0.data))
        self.lambda_euc = self.args.x0_sigma ** 2 / y0_sigma ** 2

    def create_image_plane(self):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        x_data = np.random.randn(*self.x0_data.shape).astype('f')
        x_data = x_data / np.linalg.norm(x_data) * self.args.x0_sigma
        x_data = xp.asarray(x_data, dtype=xp.float32)
        self.x_link = chainer.links.Parameter(x_data)
        # initial_img = self.deprocess(cuda.to_cpu(self.x_link.W.data)[0])
        # cv.imwrite('{}/{}_{}_init.png'.format(
        #     self.out_dir, self.img_id, self.args.layer), initial_img)

    def prepare_optimizer(self):
        if self.args.opt == 'MomentumSGD':
            self.opt = optimizers.MomentumSGD(momentum=0.9)
        elif self.args.opt == 'Adam':
            self.opt = optimizers.Adam(alpha=self.args.adam_alpha)
            print('Adam alpha=', self.args.adam_alpha)
        else:
            raise ValueError('Opt should be MomentumSGD or Adam.')
        self.opt.setup(self.x_link)

    def create_lr_schedule(self):
        self.lr_schedule = 0.0015 * np.array(
            [0.1] * 10000 +
            [0.05] * 10000 +
            [0.01] * 10000, dtype='f')

    def tvh(self, x):
        return F.convolution_2d(x, W=self.Wh)

    def tvw(self, x):
        return F.convolution_2d(x, W=self.Ww)

    def tv_norm(self, x):
        diffh = self.tvh(
            F.reshape(x, (3, 1, self.args.in_size, self.args.in_size)))
        diffw = self.tvw(
            F.reshape(x, (3, 1, self.args.in_size, self.args.in_size)))
        tv = (F.sum(diffh ** 2) + F.sum(diffw ** 2)) ** (self.args.beta / 2.)

        return tv

    def __call__(self, x):
        y = self.extract_feature(x)
        e = F.mean_squared_error(self.y0, y)
        tv = self.tv_norm(x)
        self.loss = (self.lambda_euc * float(self.y0.data.size) * e +
                     self.args.lambda_tv * tv +
                     self.args.lambda_lp * F.sum(x ** self.args.p))

        return self.loss


def write_result(args, inverter, x):
    result = inverter.deprocess(cuda.to_cpu(x.data)[0])
    if args.channels < 0:
        cv.imwrite('{}/{}_{}_result.png'.format(
            inverter.out_dir, inverter.img_id, args.layer), result)
    else:
        cv.imwrite('{}/{}_{}_{}_result.png'.format(
            inverter.out_dir, inverter.img_id, args.layer, args.channels),
            result)

if __name__ == '__main__':
    args = get_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    inverter = InvertFeature(args)

    min_error = np.finfo('f').max
    optimal_x = None
    i = 0
    while True:
        try:
            if args.opt == 'MomentumSGD':
                inverter.opt.lr = inverter.lr_schedule[i]
            x = inverter.x_link.W
            inverter.opt.update(inverter, x)

            l = cuda.to_cpu(inverter.loss.data)
            if l < min_error:
                optimal_x = x
                min_error = l
            print('{}:\tloss:{}\tmin_loss:{}'.format(i, l, min_error))

            i += 1
            if args.opt == 'MomentumSGD' and i >= len(inverter.lr_schedule):
                break
            if args.opt == 'Adam' and i == args.max_iter:
                break
        except KeyboardInterrupt:
            write_result(args, inverter, x)

    write_result(args, inverter, optimal_x)
