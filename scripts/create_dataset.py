#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
import time

import numpy as np

import cv2 as cv
import lmdb
from utils.patches import divide_to_patches

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def create_merged_map():
    # copy sat images
    for data_type in ['train', 'test', 'valid']:
        out_dir = 'data/mass_merged/%s/sat' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('data/mass_buildings/%s/sat/*.tiff' % data_type):
            shutil.copy(fn, '%s/%s' % (out_dir, os.path.basename(fn)))

    road_maps = dict([(os.path.basename(fn).split('.')[0], fn)
                      for fn in glob.glob('data/mass_roads/*/map/*.tif')])

    # combine map images
    for data_type in ['train', 'test', 'valid']:
        out_dir = 'data/mass_merged/%s/map' % data_type
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fn in glob.glob('data/mass_buildings/%s/map/*.tif' % data_type):
            base = os.path.basename(fn).split('.')[0]
            building_map = cv.imread(fn, cv.IMREAD_GRAYSCALE)
            road_map = cv.imread(road_maps[base], cv.IMREAD_GRAYSCALE)
            _, building_map = cv.threshold(
                building_map, 0, 1, cv.THRESH_BINARY)
            _, road_map = cv.threshold(road_map, 0, 1, cv.THRESH_BINARY)
            h, w = road_map.shape
            merged_map = np.zeros((h, w))
            merged_map += building_map
            merged_map += road_map * 2
            merged_map = np.where(merged_map > 2, 2, merged_map)
            cv.imwrite('data/mass_merged/%s/map/%s.tif' % (data_type, base),
                       merged_map)
            print(merged_map.shape, fn)
            merged_map = np.array([np.where(merged_map == 0, 1, 0),
                                   np.where(merged_map == 1, 1, 0),
                                   np.where(merged_map == 2, 1, 0)])
            merged_map = merged_map.swapaxes(0, 2).swapaxes(0, 1)
            cv.imwrite('data/mass_merged/%s/map/%s.png' % (data_type, base),
                       merged_map * 255)


def create_single_maps(map_data_dir):
    for map_fn in glob.glob('%s/*.tif*' % map_data_dir):
        map = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        _, map = cv.threshold(map, 0, 1, cv.THRESH_BINARY)
        cv.imwrite(map_fn, map)


def create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                   sat_data_dir, map_data_dir, sat_out_dir, map_out_dir):
    if os.path.exists(sat_out_dir):
        shutil.rmtree(sat_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    os.makedirs(sat_out_dir)
    os.makedirs(map_out_dir)

    # db
    sat_env = lmdb.Environment(sat_out_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=False)
    map_env = lmdb.Environment(map_out_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=False)

    # patch size
    sat_size = sat_patch_size
    map_size = map_patch_size
    print('patch size:', sat_size, map_size, stride)

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))
    index = np.arange(len(sat_fns))
    np.random.shuffle(index)
    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    # create keys
    keys = np.arange(15000000)
    np.random.shuffle(keys)

    n_all_files = len(sat_fns)
    print('n_all_files:', n_all_files)

    n_patches = 0
    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((os.path.basename(sat_fn).split('.')[0]) !=
                (os.path.basename(map_fn).split('.')[0])):
            print('File names are different', sat_fn, map_fn)
            return

        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        map_im = map_im[:, :, np.newaxis]
        st = time.time()
        sat_patches, map_patches = divide_to_patches(
            stride, sat_size, map_size, sat_im, map_im)
        print('divide:{}'.format(time.time() - st))
        sat_patches = np.asarray(sat_patches, dtype=np.uint8)
        map_patches = np.asarray(map_patches, dtype=np.uint8)
        for patch_i in range(sat_patches.shape[0]):
            sat_patch = sat_patches[patch_i]
            map_patch = map_patches[patch_i]
            key = b'%010d' % keys[n_patches]
            sat_txn.put(key, sat_patch.tobytes())
            map_txn.put(key, map_patch.tobytes())

            n_patches += 1

        print(file_i, '/', n_all_files, 'n_patches:', n_patches)

    sat_txn.commit()
    sat_env.close()
    map_txn.commit()
    map_env.close()
    print('patches:\t', n_patches)


def roads_mini(map_dir, sat_dir, out_map_dir, out_sat_dir):
    if os.path.exists(out_map_dir):
        shutil.rmtree(out_map_dir)
    if os.path.exists(out_sat_dir):
        shutil.rmtree(out_sat_dir)
    shutil.copytree(map_dir, out_map_dir)
    shutil.copytree(sat_dir, out_sat_dir)
    for map_fn in glob.glob('%s/*.tif*' % out_map_dir):
        base, ext = os.path.splitext(map_fn)
        png_fn = map_fn.replace(ext, '.png')
        if os.path.exists(png_fn):
            os.remove(png_fn)
        map = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)
        cv.imwrite(map_fn, np.array(map == 2, dtype=np.uint8) * 255)

if __name__ == '__main__':
    if args.dataset == 'multi':
        create_merged_map()

    if args.dataset == 'single':
        create_single_maps('data/mass_roads/valid/map')
        create_single_maps('data/mass_roads/test/map')
        create_single_maps('data/mass_roads/train/map')
        create_single_maps('data/mass_buildings/valid/map')
        create_single_maps('data/mass_buildings/test/map')
        create_single_maps('data/mass_buildings/train/map')

    if args.dataset == 'roads_mini':
        # road channel of merged dataset
        roads_mini('data/mass_merged/valid/map',
                   'data/mass_merged/valid/sat',
                   'data/mass_roads_mini/valid/map',
                   'data/mass_roads_mini/valid/sat')
        roads_mini('data/mass_merged/test/map',
                   'data/mass_merged/test/sat',
                   'data/mass_roads_mini/test/map',
                   'data/mass_roads_mini/test/sat')
        roads_mini('data/mass_merged/train/map',
                   'data/mass_merged/train/sat',
                   'data/mass_roads_mini/train/map',
                   'data/mass_roads_mini/train/sat')
        create_single_maps('data/mass_roads_mini/valid/map')
        create_single_maps('data/mass_roads_mini/test/map')
        create_single_maps('data/mass_roads_mini/train/map')
        create_patches(92, 24, 16, 1,
                       'data/mass_roads_mini/valid/sat',
                       'data/mass_roads_mini/valid/map',
                       'data/mass_roads_mini/lmdb/valid_sat',
                       'data/mass_roads_mini/lmdb/valid_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_roads_mini/test/sat',
                       'data/mass_roads_mini/test/map',
                       'data/mass_roads_mini/lmdb/test_sat',
                       'data/mass_roads_mini/lmdb/test_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_roads_mini/train/sat',
                       'data/mass_roads_mini/train/map',
                       'data/mass_roads_mini/lmdb/train_sat',
                       'data/mass_roads_mini/lmdb/train_map')

    if args.dataset == 'roads':
        create_patches(92, 24, 16, 1,
                       'data/mass_roads/valid/sat',
                       'data/mass_roads/valid/map',
                       'data/mass_roads/lmdb/valid_sat',
                       'data/mass_roads/lmdb/valid_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_roads/test/sat',
                       'data/mass_roads/test/map',
                       'data/mass_roads/lmdb/test_sat',
                       'data/mass_roads/lmdb/test_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_roads/train/sat',
                       'data/mass_roads/train/map',
                       'data/mass_roads/lmdb/train_sat',
                       'data/mass_roads/lmdb/train_map')

    if args.dataset == 'buildings':
        create_patches(92, 24, 16, 1,
                       'data/mass_buildings/valid/sat',
                       'data/mass_buildings/valid/map',
                       'data/mass_buildings/lmdb/valid_sat',
                       'data/mass_buildings/lmdb/valid_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_buildings/test/sat',
                       'data/mass_buildings/test/map',
                       'data/mass_buildings/lmdb/test_sat',
                       'data/mass_buildings/lmdb/test_map')
        create_patches(92, 24, 16, 1,
                       'data/mass_buildings/train/sat',
                       'data/mass_buildings/train/map',
                       'data/mass_buildings/lmdb/train_sat',
                       'data/mass_buildings/lmdb/train_map')

    if args.dataset == 'merged':
        create_patches(92, 24, 16, 3,
                       'data/mass_merged/valid/sat',
                       'data/mass_merged/valid/map',
                       'data/mass_merged/lmdb/valid_sat',
                       'data/mass_merged/lmdb/valid_map')
        create_patches(92, 24, 16, 3,
                       'data/mass_merged/test/sat',
                       'data/mass_merged/test/map',
                       'data/mass_merged/lmdb/test_sat',
                       'data/mass_merged/lmdb/test_map')
        create_patches(92, 24, 16, 3,
                       'data/mass_merged/train/sat',
                       'data/mass_merged/train/map',
                       'data/mass_merged/lmdb/train_sat',
                       'data/mass_merged/lmdb/train_map')
