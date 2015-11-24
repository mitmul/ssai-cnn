#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv


class Transform(object):

    def __init__(self, args):
        self.args = args

    def transform(self, ortho, label):
        self.ortho = ortho
        self.label = label

        if self.args.fliplr == 1:
            if np.random.randint(2) == 1:
                self.ortho = np.fliplr(self.ortho)
                self.label = np.fliplr(self.label)

        if self.args.rotate == 1:
            angle = np.random.randint(self.args.angle)
            ortho_h, ortho_w, _ = self.ortho.shape
            label_h, label_w = self.label.shape

            center = (ortho_w // 2, ortho_h // 2)
            r = cv.getRotationMatrix2D(center, angle, 1.0)
            self.ortho = cv.warpAffine(self.ortho, r, (ortho_w, ortho_h),
                                       flags=cv.INTER_NEAREST)

            center = (label_w // 2, label_h // 2)
            r = cv.getRotationMatrix2D(center, angle, 1.0)
            self.label = cv.warpAffine(self.label, r, (label_w, label_h),
                                       flags=cv.INTER_NEAREST)

        if self.args.crop == 1:
            oh, ow, _ = self.ortho.shape
            self.ortho = self.ortho[oh / 2 - self.args.ortho_side / 2:
                                    oh / 2 + self.args.ortho_side / 2,
                                    ow / 2 - self.args.ortho_side / 2:
                                    ow / 2 + self.args.ortho_side / 2, :]
            lh, lw = self.label.shape
            self.label = self.label[lh / 2 - self.args.label_side / 2:
                                    lh / 2 + self.args.label_side / 2,
                                    lw / 2 - self.args.label_side / 2:
                                    lw / 2 + self.args.label_side / 2]

        if self.ortho.dtype != np.float32:
            self.ortho = self.ortho.astype(np.float32)

        if self.label.dtype != np.int32:
            self.label = self.label.astype(np.int32)

        if self.args.norm == 1:
            self.ortho -= self.ortho.reshape(-1, 3).mean(axis=0)
            self.ortho /= self.ortho.reshape(-1, 3).std(axis=0) + 1e-5

        return self.ortho, self.label
