#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

epoch=400
dataset=merged
channels=3
pad=24
steps=256
offset=8

function evaluate() {
    nohup python scripts/eval_urban.py \
    --test_map_dir data/mass_$dataset/test/map \
    --result_dir $1/prediction_$epoch \
    --pad $pad \
    --offset $offset \
    --steps $steps &
}

evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.959905
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.804084
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.741999
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.734032
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.610897
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.586655
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26_1.436853
evaluate results/cis_1.0/MnihCNN_cis_2016-01-20_06-35-26
