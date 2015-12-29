#! /bin/bash

PYTHONPATH="." python tests/test_dataset.py --ortho_db data/mass_buildings/lmdb/train_sat --label_db data/mass_buildings/lmdb/train_map --out_dir data/mass_buildings/patch_test
PYTHONPATH="." python tests/test_dataset.py --ortho_db data/mass_roads/lmdb/train_sat --label_db data/mass_roads/lmdb/train_map --out_dir data/mass_roads/patch_test
PYTHONPATH="." python tests/test_dataset.py --ortho_db data/mass_roads_mini/lmdb/train_sat --label_db data/mass_roads_mini/lmdb/train_map --out_dir data/mass_roads_mini/patch_test
PYTHONPATH="." python tests/test_dataset.py --ortho_db data/mass_merged/lmdb/train_sat --label_db data/mass_merged/lmdb/train_map --out_dir data/mass_merged/patch_test
