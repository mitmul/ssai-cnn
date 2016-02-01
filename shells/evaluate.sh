#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

epoch=400
dataset=merged
channels=3
pad=24
relax=3
steps=1024

function evaluate() {
    nohup python scripts/evaluate_single.py \
    --map_dir data/mass_$dataset/test/map \
    --result_dir $1/prediction_$epoch \
    --channel $channels \
    --offset $2 \
    --pad $pad \
    --relax $relax \
    --steps $steps &
}

function evaluate_() {
    nohup python scripts/evaluate.py \
    --map_dir data/mass_$dataset/test/map \
    --result_dir $1/prediction_$epoch \
    --channel $channels \
    --offset $2 \
    --pad $pad \
    --relax $relax \
    --steps $steps &
}

evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.959905 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.804084 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.741999 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.734032 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.610897 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.586655 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.436853 1
evaluate_ results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26 1
