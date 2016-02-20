#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

model = 'cis'
epoch = 400


def get_pre_rec(pre_rec):
    pos = np.argmin((pre_rec[:, 0] - pre_rec[:, 1]) ** 2)
    return pre_rec[pos]


bldg_recalls = []
road_recalls = []

for dn in glob.glob('results/{}_*'.format(model)):
    s = re.search('/cis_([0-9\.]+)', dn)
    if s is None:
        continue
    ratio = float(s.groups()[0])
    bldg_npy = glob.glob(
        '{}/Mnih*/prediction_{}/evaluation_{}/pre_rec_1.npy'.format(
            dn, epoch, epoch))[0]
    road_npy = glob.glob(
        '{}/Mnih*/prediction_{}/evaluation_{}/pre_rec_2.npy'.format(
            dn, epoch, epoch))[0]

    bldg = np.load(bldg_npy)
    road = np.load(road_npy)

    bldg_recalls.append((ratio, get_pre_rec(bldg)[1]))
    road_recalls.append((ratio, get_pre_rec(road)[1]))

bldg_recalls = np.array(sorted(bldg_recalls))
road_recalls = np.array(sorted(road_recalls))

np.save('{}_dif_datasize_bldg'.format(model), bldg_recalls)
np.save('{}_dif_datasize_road'.format(model), road_recalls)

plt.plot(bldg_recalls[:, 0], bldg_recalls[:, 1], 'x-', label='Buildings')
plt.plot(road_recalls[:, 0], road_recalls[:, 1], 'x-', label='Roads')
plt.savefig('test.png')
