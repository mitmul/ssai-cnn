#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
from utils.evaluation import relax_precision
from utils.evaluation import relax_recall

side = 1500
relax = 1
pred = np.random.rand(side, side)
label = np.asarray(np.random.rand(side, side) > 0.8, dtype=np.int32)


def calc_prec_recall(pred, label, t):
    deno = 1.0 / 256 * t
    pred = np.asarray(pred >= deno, dtype=np.int32)
    positive = np.sum(pred == 1)
    true = np.sum(label == 1)
    prec_tp = relax_precision(pred, label, relax)
    recall_tp = relax_recall(pred, label, relax)

    if prec_tp > positive or recall_tp > true:
        print(positive, prec_tp, true, recall_tp)
        sys.exit('Calculation is wrong.')

    prec = prec_tp / float(positive) if positive > 0 else 1.0
    recall = recall_tp / float(true) if true > 0 else 1.0

    return (prec, recall)

for t in range(256):
    st = time.time()
    prec, recall = calc_prec_recall(pred, label, t)
    print(prec, recall, time.time() - st)
