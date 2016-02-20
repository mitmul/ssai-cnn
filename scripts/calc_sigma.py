#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np

from train import get_cursor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_fn', type=str)
    parser.add_argument('--side', type=int, default=92)
    args = parser.parse_args()
    side = args.side

    cur, txn, args.N = get_cursor(args.db_fn)

    i = 0
    norms = []
    while True:
        key, val = cur.item()
        patch = np.fromstring(val, dtype=np.uint8).reshape((side, side, 3))
        patch = patch.astype(np.float)
        patch -= patch.reshape((-1, 3)).mean(axis=0)
        patch /= patch.reshape((-1, 3)).std(axis=0)
        norms.append(np.linalg.norm(patch))
        ret = cur.next()
        if not ret:
            break
        print(i)
        i += 1
    fp = open('data/x0_sigma.txt', 'w')
    print(np.mean(norms), file=fp)
