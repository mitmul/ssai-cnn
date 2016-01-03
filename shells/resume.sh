#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

function resume() {
    echo $1
    CHAINER_SEED=$1 nohup python scripts/train.py \
    --seed $1 \
    --gpu $1 \
    --model $2/MnihCNN_multi.py \
    --train_ortho_db data/mass_merged/lmdb/train_sat \
    --train_label_db data/mass_merged/lmdb/train_map \
    --valid_ortho_db data/mass_merged/lmdb/valid_sat \
    --valid_label_db data/mass_merged/lmdb/valid_map \
    --resume_model $2/epoch-$3.model \
    --resume_opt $2/epoch-$3.state \
    --epoch_offset $3 \
    --lr 0.00005 \
    > mnih_multi.log 2>&1 < /dev/null &
}

resume 0 results/MnihCNN_multi_2016-01-03_07-05-05_2.518014 100
resume 2 results/MnihCNN_multi_2016-01-03_07-05-05_2.121294 100
resume 3 results/MnihCNN_multi_2016-01-03_07-05-05_2.114848 100
resume 4 results/MnihCNN_multi_2016-01-03_07-05-05_1.622673 100
resume 5 results/MnihCNN_multi_2016-01-03_07-05-05_1.500833 100
resume 6 results/MnihCNN_multi_2016-01-03_07-05-05_1.455 100
resume 7 results/MnihCNN_multi_2016-01-03_07-05-05 100
resume 8 results/MnihCNN_multi_2016-01-03_07-05-04 100
