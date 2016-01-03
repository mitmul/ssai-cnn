#! /bin/bash

function invert() {
    nohup python scripts/invert.py \
    --model $1 \
    --param $2 \
    --layer $3 \
    --gpu $4 \
    --lambda_tv $5 \
    --img_fn $6 \
    --opt MomentumSGD \
    > nohup.out 2>&1 < /dev/null &
}

function batch_invert() {
    echo $1 $2 $3
    invert $1 $2 conv1 0 0.5 $3
    invert $1 $2 relu1 2 0.5 $3
    invert $1 $2 mpool1 3 0. 5 $3
    invert $1 $2 conv2 4 0.5 $3
    invert $1 $2 relu2 5 0.5 $3
    invert $1 $2 conv3 6 1 $3
    invert $1 $2 relu3 7 5 $3
    invert $1 $2 fc4 8 5 $3
    invert $1 $2 relu4 0 5 $3
    invert $1 $2 fc5 2 5 $3
    invert $1 $2 reshape 3 5 $3
    invert $1 $2 cis 4 10 $3
    invert $1 $2 pred 5 10 $3
}

model=models/MnihCNN_cis.py
result_dir=results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30
epoch=400
param=$result_dir/epoch-$epoch.model
img_dir=data/mass_merged/trans_test

img_fns=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# img_fns=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
# img_fns=("21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
echo $img_fns
echo $param
for img_fn in ${img_fns[@]}; do
    fn=$img_dir/$img_fn.jpg
    echo $fn
    batch_invert $model $param $fn
done
