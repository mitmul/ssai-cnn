#! /bin/bash

dname=MnihCNN_multi_2015-12-18_15-06-40
epoch=120
model=MnihCNN_multi
dataset=merged
channels=3
gpu=2

python scripts/predict.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--test_sat_dir data/mass_${dataset}/test/sat \
--channels ${channels} \
--offset 8 \
--gpu ${gpu}
