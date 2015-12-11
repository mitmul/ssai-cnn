#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import chainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
