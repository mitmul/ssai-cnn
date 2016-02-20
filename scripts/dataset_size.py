#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import re
import sys

import numpy as np

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        choices=['cis', 'multi'])
    parser.add_argument('--epoch', type=int, default=400)
    return parser.parse_args()


def get_breakeven(pre_rec):
    return np.argmin((pre_rec[:, 0] - pre_rec[:, 1]) ** 2)

if __name__ == '__main__':
    args = get_args()
    plt.ylim([0.87, 1.0])

    for model_type in ['cis', 'multi']:
        bldg_recs = []
        road_recs = []
        for dname in glob.glob('results/{}_*'.format(model_type)):
            d = '{0}/integrated_{epoch}/evaluation_{epoch}'.format(
                dname, epoch=args.epoch)
            ratio = float(re.search('{}_([0-9\.]+)'.format(model_type),
                                    dname).groups()[0])
            if ratio > 0.5:
                continue
            bldg = np.load('{}/pre_rec_1.npy'.format(d))
            road = np.load('{}/pre_rec_2.npy'.format(d))
            bldg_rec = bldg[get_breakeven(bldg)][1]
            road_rec = road[get_breakeven(road)][1]
            print('[{}, {}, {}]'.format(ratio, bldg_rec, road_rec))

            bldg_recs.append((ratio, bldg_rec))
            road_recs.append((ratio, road_rec))

        bldg_recs = np.array(sorted(bldg_recs))
        road_recs = np.array(sorted(road_recs))
        plt.plot(bldg_recs[:, 0], bldg_recs[:, 1],
                 label='Building prediction ({})'.format(model_type))
        plt.plot(road_recs[:, 0], road_recs[:, 1],
                 label='Road prediction ({})'.format(model_type))

    plt.legend()
    plt.xlabel('Percentage of data used for training')
    plt.ylabel('Recall at breakeven point')
    plt.savefig('dataset_ratio.png')
