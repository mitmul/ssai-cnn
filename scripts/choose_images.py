#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re

import cv2 as cv

bldg_fns = glob.glob('data/mass_buildings/train/map/*.png')
print(len(bldg_fns))
bldg_fns += glob.glob('data/mass_buildings/valid/map/*.png')
print(len(bldg_fns))
bldg_fns += glob.glob('data/mass_buildings/test/map/*.png')
print(len(bldg_fns))
bldg_fns = set(bldg_fns)

road_fns = glob.glob('data/mass_roads/train/map/*.png')
print(len(road_fns))
road_fns += glob.glob('data/mass_roads/valid/map/*.png')
print(len(road_fns))
road_fns += glob.glob('data/mass_roads/test/map/*.png')
print(len(road_fns))
road_fns = set(road_fns)

road_mini_fns = glob.glob('data/mass_roads_mini/train/map/*.tif')
print(len(road_mini_fns))
road_mini_fns += glob.glob('data/mass_roads_mini/valid/map/*.tif')
print(len(road_mini_fns))
road_mini_fns += glob.glob('data/mass_roads_mini/test/map/*.tif')
print(len(road_mini_fns))
road_mini_fns = set(road_mini_fns)

merged_fns = glob.glob('data/mass_merged/train/map/*.png')
print(len(merged_fns))
merged_fns += glob.glob('data/mass_merged/valid/map/*.png')
print(len(merged_fns))
merged_fns += glob.glob('data/mass_merged/test/map/*.png')
print(len(merged_fns))
merged_fns = set(merged_fns)


def get_ids(fns):
    return [re.search('/([0-9]+_[0-9]+)', fn).groups()[0] for fn in fns]

road_fns = set(get_ids(road_fns))
bldg_fns = set(get_ids(bldg_fns))
road_only = road_fns.difference(bldg_fns)
print(len(road_only))
print(road_only)
print('10378675_15' in road_only)
print('10228675_15' in road_only)
print('10228735_15' in road_only)
