#! /bin/bash

function invert() {
    nohup python scripts/invert.py \
    --model $1 \
    --param $2 \
    --layer $3 \
    --gpu $4 \
    --lambda_tv $5 \
    --img_fn $6 \
    > nohup.out 2>&1 < /dev/null &
}

function batch_invert() {
    invert $model $param conv1 0 0.5 $1
    invert $model $param relu1 2 0.5 $1
    invert $model $param mpool1 3 0. 5 $1
    invert $model $param conv2 4 0.5 $1
    invert $model $param relu2 5 0.5 $1
    invert $model $param conv3 6 1 $1
    invert $model $param relu3 7 1 $1
    invert $model $param fc4 8 1 $1
    invert $model $param relu4 0 1 $1
    invert $model $param fc5 2 2 $1
    invert $model $param reshape 3 2 $1
    invert $model $param cis 4 2 $1
    invert $model $param pred 5 2 $1
}

model=models/MnihCNN_cis.py
result_dir=results
epoch=315
param=$result_dir/epoch-$epoch.model
img_dir=data/mass_merged/trans_test

img_fns=("16" "17" "22" "71" "92")
echo $img_fns
for img_fn in ${img_fns[@]}; do
    fn=$img_dir/$img_fn.jpg
    echo $fn
    batch_invert $fn
done
