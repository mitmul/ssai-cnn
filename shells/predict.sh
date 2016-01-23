#! /bin/bash

epoch=100
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

predict results/MnihCNN_cis_2016-01-20_06-35-26_1.959905 0 0
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.804084 1 2
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.741999 2 3
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.734032 3 4
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.610897 4 5
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.586655 5 6
predict results/MnihCNN_cis_2016-01-20_06-35-26_1.436853 6 7
predict results/MnihCNN_cis_2016-01-20_06-35-26 7 8

# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.568779 0 0
# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.245821 1 2
# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.85776 2 3
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.762402 3 4
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.496208 4 5
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.420262 5 6
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.372101 6 7
# predict results/MnihCNN_multi_2016-01-20_06-22-02 7 8
