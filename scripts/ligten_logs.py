#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

for fn in glob.glob('results/*/*/log.txt'):
    dname = os.path.dirname(fn)
    new_fn = fn.replace('log.txt', 'log_heavy.txt')
    os.rename(fn, new_fn)
    fp = open('{}/log.txt'.format(dname), 'w')
    for line in open(new_fn):
        if 'iter' not in line:
            print(line.strip(), file=fp)
    fp.close()
    print(fn)
