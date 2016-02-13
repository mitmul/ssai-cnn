#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

function train() {
    echo $1
    CHAINER_TYPE_CHECK=0 CHAINER_SEED=$1 \
    nohup python scripts/train.py \
    --seed $1 \
    --gpu $1 \
    --model models/MnihCNN_$2.py \
    --train_ortho_db data/mass_$3/lmdb/train_sat \
    --train_label_db data/mass_$3/lmdb/train_map \
    --valid_ortho_db data/mass_$3/lmdb/valid_sat \
    --valid_label_db data/mass_$3/lmdb/valid_map \
    --dataset_size $4 \
    > mnih_$2.log 2>&1 < /dev/null &
}

train 0 single buildings 1.0
train 5 single buildings 1.0
train 6 single buildings 1.0
train 7 single buildings 1.0
train 8 single buildings 1.0
