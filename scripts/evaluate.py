#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import imp
import glob
import time
import ctypes
import argparse
import cv2 as cv
import numpy as np
from multiprocessing import Process, Queue, Array
from chainer import cuda, serializers, Variable


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--test_map_dir', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model
    serializers.load_hdf5(args.param, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    model.train = False

    out_dir = '{}/test'.format(os.path.dirname(args.model))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for fn in glob.glob('{}/*.tif*'.format(args.test_sat_dir)):
        img = cv.imread(fn)
        pred = get_predict(args, img, model)
        out_fn = '{}/{}.png'.format(
            out_dir, os.path.splitext(os.path.basename(fn))[0])
        print(pred.shape, pred.min(), pred.max())
        import sys
        sys.exit()
        cv.imwrite(out_fn, pred * 255)
