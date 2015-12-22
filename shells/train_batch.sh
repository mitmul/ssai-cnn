#! /bin/bash

# nohup python scripts/train.py \
# --gpu 0 \
# --model models/MnihCNN_single.py \
# --train_ortho_db data/mass_buildings/lmdb/train_sat \
# --train_label_db data/mass_buildings/lmdb/train_map \
# --valid_ortho_db data/mass_buildings/lmdb/valid_sat \
# --valid_label_db data/mass_buildings/lmdb/valid_map \
# > mnih_buildings.log 2>&1 < /dev/null &
#
# nohup python scripts/train.py \
# --gpu 2 \
# --model models/MnihCNN_single.py \
# --train_ortho_db data/mass_roads/lmdb/train_sat \
# --train_label_db data/mass_roads/lmdb/train_map \
# --valid_ortho_db data/mass_roads/lmdb/valid_sat \
# --valid_label_db data/mass_roads/lmdb/valid_map \
# > mnih_roads.log 2>&1 < /dev/null &
#
# nohup python scripts/train.py \
# --gpu 3 \
# --model models/MnihCNN_single.py \
# --train_ortho_db data/mass_roads_mini/lmdb/train_sat \
# --train_label_db data/mass_roads_mini/lmdb/train_map \
# --valid_ortho_db data/mass_roads_mini/lmdb/valid_sat \
# --valid_label_db data/mass_roads_mini/lmdb/valid_map \
# > mnih_roads_mini.log 2>&1 < /dev/null &

function train() {
    echo $1
    CHAINER_SEED=$1 nohup python scripts/train.py \
    --seed $1 \
    --gpu $1 \
    --model models/MnihCNN_cis.py \
    --train_ortho_db data/mass_merged/lmdb/train_sat \
    --train_label_db data/mass_merged/lmdb/train_map \
    --valid_ortho_db data/mass_merged/lmdb/valid_sat \
    --valid_label_db data/mass_merged/lmdb/valid_map \
    > mnih_cis.log 2>&1 < /dev/null &
}

train 0
train 2
train 3
train 4
train 5
train 6
train 7
train 8

# CHAINER_SEED=1 nohup python scripts/train.py \
# --seed 1 \
# --gpu 7 \
# --model models/MnihCNN_multi.py \
# --train_ortho_db data/mass_merged/lmdb/train_sat \
# --train_label_db data/mass_merged/lmdb/train_map \
# --valid_ortho_db data/mass_merged/lmdb/valid_sat \
# --valid_label_db data/mass_merged/lmdb/valid_map \
# --resume_model results/MnihCNN_multi_2015-12-19_05-11-12/epoch-199.model \
# --resume_opt results/MnihCNN_multi_2015-12-19_05-11-12/epoch-199.state \
# --epoch_offset 199 \
# > mnih_merged.log 2>&1 < /dev/null &

# CHAINER_SEED=1 nohup python scripts/train.py \
# --gpu 8 \
# --model models/MnihCNN_cis.py \
# --train_ortho_db data/mass_merged/lmdb/train_sat \
# --train_label_db data/mass_merged/lmdb/train_map \
# --valid_ortho_db data/mass_merged/lmdb/valid_sat \
# --valid_label_db data/mass_merged/lmdb/valid_map \
# --resume_model results/MnihCNN_cis_2015-12-19_05-11-12/epoch-199.model \
# --resume_opt results/MnihCNN_cis_2015-12-19_05-11-12/epoch-199.state \
# --epoch_offset 199 \
# > mnih_merged_cis.log 2>&1 < /dev/null &

# nohup python scripts/train.py \
# --gpu 7 \
# --model models/VGG_multi.py \
# --train_ortho_db data/mass_merged/lmdb/train_sat \
# --train_label_db data/mass_merged/lmdb/train_map \
# --valid_ortho_db data/mass_merged/lmdb/valid_sat \
# --valid_label_db data/mass_merged/lmdb/valid_map \
# > mnih_merged_vgg.log 2>&1 < /dev/null &

# nohup python scripts/train.py \
# --gpu 8 \
# --model models/VGG_cis.py \
# --train_ortho_db data/mass_merged/lmdb/train_sat \
# --train_label_db data/mass_merged/lmdb/train_map \
# --valid_ortho_db data/mass_merged/lmdb/valid_sat \
# --valid_label_db data/mass_merged/lmdb/valid_map \
# > mnih_merged_vgg_cis.log 2>&1 < /dev/null &
