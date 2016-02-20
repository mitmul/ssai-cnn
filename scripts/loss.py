#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import re
import sys

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_cis', type=str)
    parser.add_argument('--log_multi', type=str)
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    return args


def get_loss(fn):
    loss = []
    for line in open(fn):
        if 'iter' not in line and 'train loss' in line:
            l = re.search('train loss:([0-9\.]+)', line.strip()).groups()[0]
            loss.append(float(l))

    return loss


if __name__ == '__main__':
    args = get_args()

    if args.result_dir is None:
        loss_cis = get_loss(args.log_cis)
        loss_multi = get_loss(args.log_multi)

        plt.plot(loss_cis, label='cis')
        plt.plot(loss_multi, label='multi')
        plt.legend()
        plt.savefig('sample.png')

    cis = None
    multi = None
    for fn in glob.glob('{}/*/log.txt'.format(args.result_dir)):
        model = re.search('MnihCNN_([a-zA-Z]+)', fn).groups()[0]
        c = 'b-' if model == 'cis' else 'r-'
        plt.plot(get_loss(fn)[:400], c)
        if model == 'cis':
            cis = get_loss(fn)[:400]
        else:
            multi = get_loss(fn)[:400]

    plt.plot(multi, 'r-', label='ours (multi-channel)')
    plt.plot(cis, 'b-', label='ours (multi-channel with CIS)')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('cross entropy loss')
    plt.savefig('loss_curve.png')
