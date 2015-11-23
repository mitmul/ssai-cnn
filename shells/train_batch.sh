#! /bin/bash

nohup python scripts/train.py \
--gpu 0 \
--model models/MnihCNN_single.py \
--train_ortho_db data/mass_buildings/lmdb/train_sat \
--train_label_db data/mass_buildings/lmdb/train_map \
--valid_ortho_db data/mass_buildings/lmdb/valid_sat \
--valid_label_db data/mass_buildings/lmdb/valid_map \
> mnih_buildings.log 2>&1 < /dev/null &

nohup python scripts/train.py \
--gpu 2 \
--model models/MnihCNN_single.py \
--train_ortho_db data/mass_roads/lmdb/train_sat \
--train_label_db data/mass_roads/lmdb/train_map \
--valid_ortho_db data/mass_roads/lmdb/valid_sat \
--valid_label_db data/mass_roads/lmdb/valid_map \
> mnih_roads.log 2>&1 < /dev/null &

nohup python scripts/train.py \
--gpu 3 \
--model models/MnihCNN_single.py \
--train_ortho_db data/mass_roads_mini/lmdb/train_sat \
--train_label_db data/mass_roads_mini/lmdb/train_map \
--valid_ortho_db data/mass_roads_mini/lmdb/valid_sat \
--valid_label_db data/mass_roads_mini/lmdb/valid_map \
> mnih_roads_mini.log 2>&1 < /dev/null &

nohup python scripts/train.py \
--gpu 4 \
--model models/MnihCNN_multi.py \
--train_ortho_db data/mass_merged/lmdb/train_sat \
--train_label_db data/mass_merged/lmdb/train_map \
--valid_ortho_db data/mass_merged/lmdb/valid_sat \
--valid_label_db data/mass_merged/lmdb/valid_map \
> mnih_merged.log 2>&1 < /dev/null &

nohup python scripts/train.py \
--gpu 5 \
--model models/MnihCNN_cis.py \
--train_ortho_db data/mass_merged/lmdb/train_sat \
--train_label_db data/mass_merged/lmdb/train_map \
--valid_ortho_db data/mass_merged/lmdb/valid_sat \
--valid_label_db data/mass_merged/lmdb/valid_map \
> mnih_merged_cis.log 2>&1 < /dev/null &
