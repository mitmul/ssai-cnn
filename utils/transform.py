#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import rotate


class Transform(object):

    def __init__(self, **params):
        [setattr(self, key, value) for key, value in params.items()]

    def transform(self, ortho, label):
        self._ortho = ortho
        self._label = label

        if hasattr(self, 'fliplr'):
            self.random_fliplr()

        if hasattr(self, 'rotate') and hasattr(self, 'angle'):
            self.random_rotate()

        if hasattr(self, 'clip'):
            if not hasattr(self, 'ortho_side'):
                raise Exception('The ortho_side attribute is needed.')
            if not hasattr(self, 'label_side'):
                raise Exception('The label_side attribute is needed.')
            self.clip_center()

        if not self._ortho.dtype == np.float32:
            self._ortho = self._ortho.astype(np.float32)

        if not self._label.dtype == np.int32:
            self._label = np.asarray(self._label > 0, dtype=np.int32)

        if hasattr(self, 'norm') and self.norm:
            # global contrast normalization
            for ch in range(self._ortho.shape[2]):
                im = self._ortho[:, :, ch]
                im = (im - np.mean(im)) / \
                    (np.std(im) + np.finfo(np.float32).eps)
                self._ortho[:, :, ch] = im

        return self._ortho, self._label

    def random_fliplr(self):
        if np.random.randint(2) == 1 and self.fliplr:
            self._ortho = np.fliplr(self._ortho)
            self._label = np.fliplr(self._label)

    def random_rotate(self):
        angle = np.random.randint(self.angle)
        self._ortho = rotate(self._ortho, angle)
        self._label = rotate(self._label, angle)

    def clip_center(self):
        oh, ow, oc = self._ortho.shape
        self._ortho = self._ortho[oh / 2 - self.ortho_side / 2:
                                  oh / 2 + self.ortho_side / 2,
                                  ow / 2 - self.ortho_side / 2:
                                  ow / 2 + self.ortho_side / 2, :]
        lh, lw, lc = self._label.shape
        self._label = self._label[lh / 2 - self.label_side / 2:
                                  lh / 2 + self.label_side / 2,
                                  lw / 2 - self.label_side / 2:
                                  lw / 2 + self.label_side / 2, :]
