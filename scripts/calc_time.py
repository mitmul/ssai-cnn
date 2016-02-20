#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import re
from datetime import datetime

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='results')
args = parser.parse_args()


def str_to_time(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S,%f')


def get_elapsed_time(log_fn):
    start_time = None
    end_time = None
    epochs = []
    for line in open(log_fn):
        if 'start training...' in line:
            start_time = line.split(' [INFO]')[0].strip()
        if 'epoch:400' in line:
            end_time = line.split(' [INFO]')[0].strip()
        if 'epoch:' in line:
            epochs.append(str_to_time(line.split(' [INFO]')[0].strip()))

    epoch_times = []
    for i in range(1, len(epochs) - 1):
        diff = epochs[i] - epochs[i - 1]
        epoch_times.append(diff.total_seconds())
    epoch_times = np.array(epoch_times)
    mean_minutes = np.mean(epoch_times) / 60

    start_time = str_to_time(start_time)
    end_time = str_to_time(end_time)
    all_time = end_time - start_time

    return mean_minutes, all_time

cis_minutes = []
cis_whole_time = []
multi_minutes = []
multi_whole_time = []
for fn in glob.glob('{}/MnihCNN_*/log.txt'.format(args.result_dir)):
    model = re.search('MnihCNN_([a-z]+)_', fn).groups()[0]
    m, w = get_elapsed_time(fn)
    if model == 'cis':
        cis_minutes.append(m)
        cis_whole_time.append(w.total_seconds())
    if model == 'multi':
        multi_minutes.append(m)
        multi_whole_time.append(w.total_seconds())

print(np.mean(cis_minutes))
print(np.mean(cis_whole_time) / 60 / 60)
print(np.mean(multi_minutes))
print(np.mean(multi_whole_time) / 60 / 60)
