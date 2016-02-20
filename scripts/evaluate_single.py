#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script intends to be used for evaluation of a single model withoug MA
# but the predictions are created with offset


import argparse
import ctypes
import glob
import os
import re
import sys
from multiprocessing import Array
from multiprocessing import Process
from multiprocessing import Queue
from os.path import basename
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import six

import cv2 as cv
from utils.evaluation import relax_precision
from utils.evaluation import relax_recall

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')


parser = argparse.ArgumentParser()
parser.add_argument('--map_dir', type=str)
parser.add_argument('--result_dir', type=str)
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--offset', type=int, default=8)
parser.add_argument('--pad', type=int, default=24)  # (64 / 2) - (16 / 2)
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--relax', type=int, default=3)
parser.add_argument('--n_thread', type=int, default=8)
args = parser.parse_args()
print(args)

result_dir = args.result_dir
n_iter = int(result_dir.split('_')[-1])
label_dir = args.map_dir
result_fns = sorted(glob.glob('%s/*.npy' % result_dir))
n_results = len(result_fns)
eval_dir = '%s/evaluation_%d' % (result_dir, n_iter)

all_positive_base = Array(
    ctypes.c_double, n_results * args.channel * args.steps)
all_positive = np.ctypeslib.as_array(all_positive_base.get_obj())
all_positive = all_positive.reshape((n_results, args.channel, args.steps))

all_prec_tp_base = Array(
    ctypes.c_double, n_results * args.channel * args.steps)
all_prec_tp = np.ctypeslib.as_array(all_prec_tp_base.get_obj())
all_prec_tp = all_prec_tp.reshape((n_results, args.channel, args.steps))

all_true_base = Array(
    ctypes.c_double, n_results * args.channel * args.steps)
all_true = np.ctypeslib.as_array(all_true_base.get_obj())
all_true = all_true.reshape((n_results, args.channel, args.steps))

all_recall_tp_base = Array(
    ctypes.c_double, n_results * args.channel * args.steps)
all_recall_tp = np.ctypeslib.as_array(all_recall_tp_base.get_obj())
all_recall_tp = all_recall_tp.reshape((n_results, args.channel, args.steps))


def makedirs(dname):
    if not exists(dname):
        os.makedirs(dname)


def get_pre_rec(positive, prec_tp, true, recall_tp, steps):
    pre_rec = []
    breakeven = []
    for t in six.moves.range(steps):
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


def draw_pre_rec_curve(pre_rec, breakeven_pt):
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


def worker_thread(result_fn_queue):
    while True:
        i, result_fn = result_fn_queue.get()
        if result_fn is None:
            break

        img_id = basename(result_fn).split('pred_')[-1]
        img_id, _ = os.path.splitext(img_id)
        if '.' in img_id:
            img_id = img_id.split('.')[0]
        if len(re.findall('_', img_id)) > 1:
            img_id = '_'.join(img_id.split('_')[1:])
        out_dir = '%s/%s' % (eval_dir, img_id)
        makedirs(out_dir)
        print(img_id)

        label = cv.imread('%s/%s.tif' %
                          (label_dir, img_id), cv.IMREAD_GRAYSCALE)
        pred = np.load(result_fn)
        pred = pred[args.offset:1500 - (60 - args.offset),
                    args.offset:1500 - (60 - args.offset)]
        label = label[args.pad + args.offset:
                      args.pad + args.offset + pred.shape[0],
                      args.pad + args.offset:
                      args.pad + args.offset + pred.shape[1]]
        cv.imwrite('%s/label_%s.png' % (out_dir, img_id), label * 125)

        print('label_shape:', label.shape)
        print('pred_shape:', pred.shape)

        for c in six.moves.range(args.channel):
            for t in six.moves.range(0, args.steps):
                threshold = 1.0 / args.steps * t

                pred_vals = np.array(
                    pred[:, :, c] >= threshold, dtype=np.int32)

                label_vals = np.array(label, dtype=np.int32)
                if args.channel > 1:
                    label_vals = np.array(label == c, dtype=np.int32)

                all_positive[i, c, t] = np.sum(pred_vals)
                all_prec_tp[i, c, t] = relax_precision(
                    pred_vals, label_vals, args.relax)

                all_true[i, c, t] = np.sum(label_vals)
                all_recall_tp[i, c, t] = relax_recall(
                    pred_vals, label_vals, args.relax)

            pre_rec, breakeven_pt = get_pre_rec(
                all_positive[i, c], all_prec_tp[i, c],
                all_true[i, c], all_recall_tp[i, c], args.steps)

            draw_pre_rec_curve(pre_rec, breakeven_pt)
            plt.savefig('%s/pr_curve_%d.png' % (out_dir, c))
            np.save('%s/pre_rec_%d' % (out_dir, c), pre_rec)
            cv.imwrite('%s/pred_%d.png' % (out_dir, c), pred[:, :, c] * 255)

            print(img_id, c, breakeven_pt)
    print('thread finished')


if __name__ == '__main__':
    result_fn_queue = Queue()
    workers = [Process(target=worker_thread,
                       args=(result_fn_queue,)) for i in range(args.n_thread)]
    for w in workers:
        w.start()
    [result_fn_queue.put((i, fn)) for i, fn in enumerate(result_fns)]
    [result_fn_queue.put((None, None)) for _ in range(args.n_thread)]
    for w in workers:
        w.join()
    print('all finished')

    all_positive = np.sum(all_positive, axis=0)
    all_prec_tp = np.sum(all_prec_tp, axis=0)
    all_true = np.sum(all_true, axis=0)
    all_recall_tp = np.sum(all_recall_tp, axis=0)
    for c in six.moves.range(args.channel):
        pre_rec, breakeven_pt = get_pre_rec(
            all_positive[c], all_prec_tp[c],
            all_true[c], all_recall_tp[c], args.steps)
        draw_pre_rec_curve(pre_rec, breakeven_pt)
        plt.savefig('%s/pr_curve_%d.png' % (eval_dir, c))
        np.save('%s/pre_rec_%d' % (eval_dir, c), pre_rec)

        print('result:', c, breakeven_pt)
