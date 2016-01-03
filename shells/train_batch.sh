#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

function train() {
    echo $1
    CHAINER_TYPE_CHECK=0 CHAINER_SEED=$1 \
    nohup python scripts/train.py \
    --seed $1 \
    --gpu $1 \
    --model models/MnihCNN_$2.py \
    --train_ortho_db data/mass_merged/lmdb/train_sat \
    --train_label_db data/mass_merged/lmdb/train_map \
    --valid_ortho_db data/mass_merged/lmdb/valid_sat \
    --valid_label_db data/mass_merged/lmdb/valid_map \
    --dataset_size $3 \
    > mnih_$2.log 2>&1 < /dev/null &
}

train 0 cis 0.3
train 2 cis 0.3
train 3 cis 0.3
train 4 cis 0.3
train 5 cis 0.3
train 6 cis 0.3
train 7 cis 0.3
train 8 cis 0.3
