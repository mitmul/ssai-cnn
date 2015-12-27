#! /bin/bash

function invert() {
    nohup python scripts/invert.py --model $1 --param $2 --layer $3 --gpu $4 \
    > nohup.out 2>&1 < /dev/null &
}

model=models/MnihCNN_cis.py
result_dir=results/cis/MnihCNN_cis_2015-12-21_12-48-31
epoch=300
param=$result_dir/epoch-$epoch.model

invert $model $param conv1 0
invert $model $param relu1 2
invert $model $param mpool1 3
invert $model $param conv2 4
invert $model $param relu2 5
invert $model $param conv3 6
invert $model $param relu3 7
invert $model $param fc4 8
invert $model $param relu4 0
invert $model $param fc5 2
invert $model $param reshape 3
invert $model $param cis 4
invert $model $param pred 5
