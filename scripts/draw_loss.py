#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import sys

import numpy as np

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def draw_loss(logfile, outfile):
    train_epoch_loss = []
    valid_epoch_loss = []
    for line in open(logfile):
        line = line.strip()
        if 'epoch:' not in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        if 'iter' not in line and 'train loss' in line:
            tr_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
            train_epoch_loss.append([epoch, tr_l])
        if 'iter' not in line and 'validate loss' in line:
            va_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
            valid_epoch_loss.append([epoch, va_l])

    train_epoch_loss = np.asarray(train_epoch_loss)
    valid_epoch_loss = np.asarray(valid_epoch_loss)

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_epoch_loss[:, 0], train_epoch_loss[:, 1], c='b',
             label='training loss', marker='x')

    if valid_epoch_loss.shape[0] > 2:
        plt.plot(valid_epoch_loss[:, 0], valid_epoch_loss[:, 1], c='r',
                 label='validation loss', marker='x')

    plt.legend(loc='upper right')
    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--outfile', type=str, default='log.png')
    args = parser.parse_args()

    draw_loss(args.logfile, args.outfile)
