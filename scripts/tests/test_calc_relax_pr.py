#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'scripts/utils')
import time
import numpy as np
from calc_relax_pr import relax_precision, relax_recall

side = 1500
relax = 1
pred = np.random.rand(side, side)
label = np.asarray(np.random.rand(side, side) > 0.8, dtype=np.uint8)


def calc_prec_recall(pred, label, t):
    deno = 1.0 / 256 * t
    p = np.asarray(pred >= deno, dtype=np.uint8)
    posi = np.sum(p)
    true = np.sum(label)
    pr = relax_precision(p, label, relax)
    rc = relax_recall(p, label, relax)
    prec = pr / float(posi) if posi > 0 else 1.0
    recall = rc / float(true) if true > 0 else 1.0

    print(np.sum(p == label) / float(posi), np.sum(p == label) / float(true))

    return (prec, recall)

for t in range(256):
    st = time.time()
    prec, recall = calc_prec_recall(pred, label, t)
    print(prec, recall, time.time() - st)
