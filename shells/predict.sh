#! /bin/bash

dname=MnihCNN_cis_2015-11-27_20-23-27
epoch=84
model=MnihCNN_cis
dataset=merged
channels=3

python scripts/predict.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--test_sat_dir data/mass_${dataset}/test/sat \
--channels ${channels}
