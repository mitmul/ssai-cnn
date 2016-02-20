#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv
from tqdm import tqdm
from utils.evaluation import relax_precision
from utils.evaluation import relax_recall

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')


PATCH_SIZE = 16
PATCH_PIXELS = PATCH_SIZE ** 2
STRIDE = 16
NUM_RATIO = 1.0 / 3.0
RELAX = 3


def get_relaxed_pre_rec(p_patch, l_patch):
    p_patch = np.array(p_patch, dtype=np.int32)
    l_patch = np.array(l_patch, dtype=np.int32)

    positive = np.sum(p_patch == 1)
    prec_tp = relax_precision(p_patch, l_patch, RELAX)
    true = np.sum(l_patch == 1)
    recall_tp = relax_recall(p_patch, l_patch, RELAX)

    if prec_tp > positive or recall_tp > true:
        print(positive, prec_tp, true, recall_tp)
        sys.exit('Calculation is wrong.')

    return positive, prec_tp, true, recall_tp


def get_pre_rec(positive, prec_tp, true, recall_tp, steps):
    pre_rec = []
    breakeven = []
    for t in range(steps):
        if positive[t] < prec_tp[t] or true[t] < recall_tp[t]:
            sys.exit('calculation is wrong')
        pre = float(prec_tp[t]) / positive[t] if positive[t] > 0 else 0
        rec = float(recall_tp[t]) / true[t] if true[t] > 0 else 0
        pre_rec.append([pre, rec])
        if pre != 1 and rec != 1 and pre > 0 and rec > 0:
            breakeven.append([pre, rec])
    pre_rec = np.asarray(pre_rec)
    breakeven = np.asarray(breakeven)
    breakeven_pt = np.abs(breakeven[:, 0] - breakeven[:, 1]).argmin()
    breakeven_pt = breakeven[breakeven_pt]

    return pre_rec, breakeven_pt


def get_complex_regions(args, label_fn, pred_fns):
    fn = re.search('(.+)\.tif', os.path.basename(label_fn)).groups()[0]
    pred = np.load(pred_fns[fn])
    label = cv.imread(label_fn, cv.IMREAD_GRAYSCALE)
    label = label[args.pad + args.offset - 1:
                  args.pad + args.offset - 1 + pred.shape[0],
                  args.pad + args.offset - 1:
                  args.pad + args.offset - 1 + pred.shape[1]]
    if pred.shape[2] == 1:
        pred = pred[:, :, 0]
        pred = np.array([pred, pred, pred]).transpose(1, 2, 0)
    print('pred.shape:', pred.shape)

    thresh_evals = []
    for thresh in tqdm(range(args.steps)):
        pred_th = np.zeros(pred.shape, dtype=np.int32)
        th = thresh / float(args.steps - 1)
        for ch in range(pred.shape[2]):
            pred_th[:, :, ch] = np.array(pred[:, :, ch] >= th, dtype=np.int32)

        patch_evals = []
        for y in range(0, label.shape[0], STRIDE):
            for x in range(0, label.shape[1], STRIDE):
                if (y + PATCH_SIZE) >= label.shape[0]:
                    y = label.shape[0] - PATCH_SIZE
                if (x + PATCH_SIZE) >= label.shape[1]:
                    x = label.shape[1] - PATCH_SIZE

                l_patch = label[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                bgnd_ch = np.array(l_patch == 0, dtype=np.int32)
                bldg_ch = np.array(l_patch == 1, dtype=np.int32)
                road_ch = np.array(l_patch == 2, dtype=np.int32)
                l_patch = [bgnd_ch, bldg_ch, road_ch]

                num_bldg_pix = np.sum(bldg_ch)
                num_road_pix = np.sum(road_ch)
                if ((num_bldg_pix > (PATCH_PIXELS * NUM_RATIO)) and
                        (num_road_pix > (PATCH_PIXELS * NUM_RATIO))):
                    region_eval = []
                    for ch in range(pred.shape[2]):
                        p = pred_th[y:y + PATCH_SIZE, x:x + PATCH_SIZE, ch]
                        rpr = get_relaxed_pre_rec(p, l_patch[ch])
                        region_eval.append(list(rpr))
                    patch_evals.append(region_eval)
        evals = np.zeros((pred.shape[2], 4))
        for r in patch_evals:
            evals += np.array(r)
        for ch in range(pred.shape[2]):
            positive, prec_tp, true, recall_tp = evals[ch]
            pre = float(prec_tp) / positive if positive > 0 else 0
            rec = float(recall_tp) / true if true > 0 else 0
            if pre > 1.0 or rec > 1.0:
                print('{}({}):{:.4f} - {:.4f}'.format(ch, thresh, pre, rec))
                sys.exit('Calculation is wrong.')
        thresh_evals.append(evals)
    thresh_evals = np.array(thresh_evals)

    return thresh_evals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--test_map_dir', type=str)
    parser.add_argument('--pad', type=int, default=24)
    parser.add_argument('--offset', type=int, default=8)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()

    pred_fns = []
    for result in glob.glob('{}/*.npy'.format(args.result_dir)):
        fn = re.search('(.+)\.npy', os.path.basename(result)).groups()[0]
        pred_fns.append((fn, result))
    pred_fns = dict(pred_fns)

    args.out_dir = '{}/urban_{}'.format(args.result_dir, NUM_RATIO)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # threshold, channels, (positive, prec_tp, true, recall_tp)
    # n_ch = np.load(list(pred_fns.items())[0][1]).shape[2]
    n_ch = 3
    evals = np.zeros((args.steps, n_ch, 4))
    for label_fn in glob.glob('{}/*.tif*'.format(args.test_map_dir)):
        evals += get_complex_regions(args, label_fn, pred_fns)
        print(label_fn)
    np.save('{}/evals'.format(args.out_dir), evals)

    for ch in range(n_ch):
        e = evals[:, ch, :]
        pre_rec, breakeven_pt = \
            get_pre_rec(e[:, 0], e[:, 1], e[:, 2], e[:, 3], args.steps)

        plt.clf()
        plt.plot(pre_rec[:, 0], pre_rec[:, 1])
        plt.plot(breakeven_pt[0], breakeven_pt[1],
                 'x', label='breakeven recall: %f' % (breakeven_pt[1]))
        plt.ylabel('recall')
        plt.xlabel('precision')
        plt.ylim([0.0, 1.1])
        plt.xlim([0.0, 1.1])
        plt.legend(loc='lower left')
        plt.grid(linestyle='--')
        plt.savefig('{}/pre_rec_{}.png'.format(args.out_dir, ch))
