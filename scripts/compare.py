#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_rec_A_dir', type=str,
                        default='results/cis/integrated_100/evaluation_100')
    parser.add_argument('--pre_rec_B_dir', type=str,
                        default='results/multi/integrated_100/evaluation_100')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    epoch = int(
        re.search('evaluation_([0-9]+)', args.pre_rec_A_dir).groups()[0])

    bldg_A = np.load('{}/pre_rec_1.npy'.format(args.pre_rec_A_dir))
    bldg_B = np.load('{}/pre_rec_1.npy'.format(args.pre_rec_B_dir))
    road_A = np.load('{}/pre_rec_2.npy'.format(args.pre_rec_A_dir))
    road_B = np.load('{}/pre_rec_2.npy'.format(args.pre_rec_B_dir))

    plt.plot(bldg_A[:, 0], bldg_A[:, 1], label='cis (buildings)')
    plt.plot(bldg_B[:, 0], bldg_B[:, 1], label='multi (buildings)')
    plt.plot(road_A[:, 0], road_A[:, 1], label='cis (roads)')
    plt.plot(road_B[:, 0], road_B[:, 1], label='multi (roads)')
    plt.xlim([0.88, 0.98])
    plt.ylim([0.88, 0.98])
    plt.legend()
    plt.savefig('sample.png')
