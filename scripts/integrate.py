#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import cv2 as cv
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    fns = glob.glob(
        '{}/*/prediction_{}/*.npy'.format(args.result_dir, args.epoch))
    preds = defaultdict(list)
    for fn in fns:
        dname = fn.split('/')[1]
        imname = os.path.basename(fn).split('.')[0]
        preds[imname].append(fn)

    pred_npys = {}
    for imname, pred_fns in preds.items():
        for pred_fn in pred_fns:
            pred = np.load(pred_fn)
            if imname not in pred_npys:
                pred_npys[imname] = pred
            else:
                pred_npys[imname] += pred

    out_dir = '{}/integrated_{}'.format(args.result_dir, args.epoch)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for imname in pred_npys.keys():
        pred_npys[imname] /= len(preds[imname])
        np.save('{}/{}'.format(out_dir, imname), pred_npys[imname])
        cv.imwrite('{}/{}.jpg'.format(out_dir, imname),
                   pred_npys[imname] * 255)
