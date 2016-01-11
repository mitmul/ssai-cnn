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

predict results/MnihCNN_cis_2016-01-04_07-05-13_2.773944 0 0
predict results/MnihCNN_cis_2016-01-04_07-05-13_2.128675 1 2
predict results/MnihCNN_cis_2016-01-04_07-05-13_2.047947 2 3
predict results/MnihCNN_cis_2016-01-04_07-05-13_2.009506 3 4
predict results/MnihCNN_cis_2016-01-04_07-05-13_1.633745 4 5
predict results/MnihCNN_cis_2016-01-04_07-05-13 5 6
predict results/MnihCNN_cis_2016-01-04_07-05-12 6 7
predict results/MnihCNN_cis_2016-01-04_07-05-11 7 8
