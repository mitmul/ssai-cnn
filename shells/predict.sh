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

epoch=150
model=MnihCNN_multi
dataset=merged
channels=3
gpu=5

# predict MnihCNN_multi_2015-12-21_10-45-53 $epoch $model $dataset $channels 0
predict MnihCNN_multi_2015-12-21_10-45-53_10 $epoch $model $dataset $channels 2
predict MnihCNN_multi_2015-12-21_10-45-53_40 $epoch $model $dataset $channels 3
predict MnihCNN_multi_2015-12-21_10-45-53_44 $epoch $model $dataset $channels 4
predict MnihCNN_multi_2015-12-21_10-45-53_46 $epoch $model $dataset $channels 5
predict MnihCNN_multi_2015-12-21_10-45-53_47 $epoch $model $dataset $channels 6
predict MnihCNN_multi_2015-12-21_10-45-53_67 $epoch $model $dataset $channels 7
predict MnihCNN_multi_2015-12-21_10-45-53_99 $epoch $model $dataset $channels 8
