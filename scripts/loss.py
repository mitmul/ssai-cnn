#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_cis', type=str)
    parser.add_argument('--log_multi', type=str)
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
    loss_cis = get_loss(args.log_cis)
    loss_multi = get_loss(args.log_multi)

    plt.plot(loss_cis, label='cis')
    plt.plot(loss_multi, label='multi')
    plt.legend()
    plt.savefig('sample.png')
