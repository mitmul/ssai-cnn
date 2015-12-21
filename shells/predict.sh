#! /bin/bash

dname=MnihCNN_cis_2015-12-19_05-11-12
epoch=200
model=MnihCNN_cis
dataset=merged
channels=3
gpu=5

python scripts/predict.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--test_sat_dir data/mass_${dataset}/test/sat \
--channels ${channels} \
--offset 8 \
--gpu ${gpu}
