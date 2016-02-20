#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import numpy as np

import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir_A', type=str)
    parser.add_argument('--result_dir_B', type=str)
    parser.add_argument('--epoch', type=int, default=300)
    args = parser.parse_args()

    out_dir = 'reshape-{}'.format(args.epoch)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fn_A = [fn for fn in sorted(
        glob.glob('{}/mid-{}/*_reshape.png'.format(
            args.result_dir_A, args.epoch)))]
    fn_B = [fn for fn in sorted(
        glob.glob('{}/mid-{}/*_reshape.png'.format(
            args.result_dir_B, args.epoch)))]

    for a, b in zip(fn_A, fn_B):
        if not os.path.basename(a) == os.path.basename(b):
            print(a, b)
            continue
        ima = cv.imread(a, cv.IMREAD_GRAYSCALE)
        imb = cv.imread(b, cv.IMREAD_GRAYSCALE)
        img = np.hstack([ima, imb])
        cv.imwrite('{}/{}'.format(out_dir, os.path.basename(a)), img)
