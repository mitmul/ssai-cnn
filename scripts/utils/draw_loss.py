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


def draw_loss(logfile, outfile):
    train_loss = []
    valid_loss = []
    for line in open(logfile):
        line = line.strip()
        if 'loss=' not in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        if 'train' in line and 'inf' not in line:
            tr_l = float(re.search('loss=([0-9\.e-]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])
        if 'valid' in line and 'inf' not in line:
            te_l = float(re.search('loss=([0-9\.e-]+)', line).groups()[0])
            valid_loss.append([epoch, te_l])

    train_loss = np.asarray(train_loss)
    valid_loss = np.asarray(valid_loss)

    plt.clf()
    fig, ax1 = plt.subplots()
    plt.plot(train_loss[:, 0], train_loss[:, 1], label='training loss')
    plt.plot(valid_loss[:, 0], valid_loss[:, 1], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.legend(loc='upper right')
    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--outfile', type=str, default='log.png')
    args = parser.parse_args()

    draw_loss(args.logfile, args.outfile)
