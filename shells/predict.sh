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

predict results/MnihCNN_cis_2016-01-11_05-16-47 0 0
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.382364 1 2
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.480355 2 3
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.496888 3 4
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.551013 4 5
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.645192 5 6
predict results/MnihCNN_cis_2016-01-11_05-16-47_1.709271 6 7
predict results/MnihCNN_cis_2016-01-11_05-16-47_2.108314 7 8
