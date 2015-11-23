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


def draw_loss(logfile, outfile, draw_iters=False):
    train_epoch_loss = []
    train_iter_loss = []
    valid_epoch_loss = []
    valid_iter_loss = []
    for line in open(logfile):
        line = line.strip()
        if 'epoch:' not in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        if 'iter' in line:
            n_iter = int(re.search('iter:([0-9]+)', line).groups()[0])
            if 'train' in line and 'inf' not in line:
                tr_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
                train_iter_loss.append([epoch, n_iter - 1, tr_l])
            if 'valid' in line and 'inf' not in line:
                te_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
                valid_iter_loss.append([epoch, n_iter - 1, te_l])
        else:
            epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
            if 'train' in line and 'inf' not in line:
                tr_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
                train_epoch_loss.append([epoch, tr_l])
            if 'valid' in line and 'inf' not in line:
                te_l = float(re.search('loss:([0-9\.e-]+)', line).groups()[0])
                valid_epoch_loss.append([epoch, te_l])

    train_iter_loss = np.asarray(train_iter_loss)
    train_iter_loss[:, 1] /= float(np.max(train_iter_loss[:, 1]))
    train_iter_loss[:, 1] += train_iter_loss[:, 0] - 1
    train_epoch_loss = np.asarray(train_epoch_loss)
    valid_iter_loss = np.asarray(valid_iter_loss)
    valid_iter_loss[:, 1] /= float(np.max(valid_iter_loss[:, 1]))
    valid_iter_loss[:, 1] += valid_iter_loss[:, 0] - 1
    valid_epoch_loss = np.asarray(valid_epoch_loss)

    plt.clf()
    if draw_iters == 1:
        plt.plot(train_iter_loss[:, 1], train_iter_loss[:, 2])
        plt.plot(valid_iter_loss[:, 1], valid_iter_loss[:, 2])
    plt.plot(train_epoch_loss[:, 0], train_epoch_loss[:, 1], c='b',
             label='training loss', marker='x')
    plt.plot(valid_epoch_loss[:, 0], valid_epoch_loss[:, 1], c='g',
             label='validation loss', marker='x')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.legend(loc='upper right')
    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--outfile', type=str, default='log.png')
    parser.add_argument('--draw_iters', type=int, default=0)
    args = parser.parse_args()

    draw_loss(args.logfile, args.outfile, args.draw_iters)
