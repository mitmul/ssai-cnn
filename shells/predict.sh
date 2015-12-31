#! /bin/bash

function predict() {
    nohup python scripts/predict.py \
    --model results/$1/$3.py \
    --param results/$1/epoch-$2.model \
    --test_sat_dir data/mass_$4/test/sat \
    --channels $5 \
    --offset 8 \
    --gpu $6 &
}

epoch=400
model=MnihCNN_cis
dataset=merged
channels=3

function predict_(){
    predict $1 $epoch $model $dataset $channels $2
}

predict_ MnihCNN_cis_2015-12-28_04-40-30 0
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.554376 2
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.609452 3
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.670797 4
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.770513 5
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.876234 6
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.88174 7
predict_ MnihCNN_cis_2015-12-28_04-40-30_3.295603 8
