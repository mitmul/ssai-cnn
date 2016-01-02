#! /bin/bash

epoch=400
model=MnihCNN_cis
dataset=merged
channels=3

function predict() {
    nohup python scripts/predict_offset.py \
    --model $1/$model.py \
    --param $1/epoch-$epoch.model \
    --test_sat_dir data/mass_$dataset/test/sat \
    --channels $channels \
    --offset $2 \
    --gpu $3 &
}


predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_3.295603 0 0
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.876234 1 2
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.770513 2 3
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.670797 3 4
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.609452 4 5
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.554376 5 6
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.88174 6 7
predict results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30 7 8
