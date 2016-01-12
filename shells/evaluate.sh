#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

epoch=200
dataset=merged
channels=3
pad=24
relax=3
steps=256

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

evaluate results/MnihCNN_cis_2016-01-11_05-16-47 0
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.382364 1
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.480355 2
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.496888 3
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.551013 4
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.645192 5
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_1.709271 6
evaluate results/MnihCNN_cis_2016-01-11_05-16-47_2.108314 7
