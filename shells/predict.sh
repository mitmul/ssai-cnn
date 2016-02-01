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

function predict_() {
    nohup python scripts/predict.py \
    --model $1/$model.py \
    --param $1/epoch-$epoch.model \
    --test_sat_dir data/mass_$dataset/test/sat \
    --channels $channels \
    --offset $2 \
    --gpu $3 &
}

# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_2.568779 8 0
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_2.245821 8 2
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_2.85776 8 3
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_1.762402 8 4
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_1.496208 8 5
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_1.420262 8 6
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02_1.372101 8 7
# predict_ results/multi_1.0/MnihCNN_multi_2016-01-20_06-22-02 8 8

predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.959905 1 0
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.804084 1 2
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.741999 1 3
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.734032 1 4
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.610897 1 5
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.586655 1 6
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.436853 1 7
predict_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26 1 8

# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_2.692095 8 0
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_2.415784 8 2
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_2.078326 8 3
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_1.965055 8 4
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_1.690456 8 5
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19_1.632728 8 6
# predict_ results/MnihCNN_cis_2016-01-29_17-30-19 8 7
# predict_ results/MnihCNN_cis_2016-01-29_17-30-18 8 8
#
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_2.967918 8 0
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_2.641496 8 2
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_2.274751 8 3
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_1.931579 8 4
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_1.614765 8 5
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_1.314742 8 6
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21_1.92536 8 7
# predict_ results/MnihCNN_multi_2016-01-29_17-36-21 8 8

# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.959905 0 0
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.804084 1 2
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.741999 2 3
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.734032 3 4
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.610897 4 5
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.586655 5 6
# predict results/MnihCNN_cis_2016-01-20_06-35-26_1.436853 6 7
# predict results/MnihCNN_cis_2016-01-20_06-35-26 7 8

# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.568779 0 0
# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.245821 1 2
# predict results/MnihCNN_multi_2016-01-20_06-22-02_2.85776 2 3
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.762402 3 4
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.496208 4 5
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.420262 5 6
# predict results/MnihCNN_multi_2016-01-20_06-22-02_1.372101 6 7
# predict results/MnihCNN_multi_2016-01-20_06-22-02 7 8
