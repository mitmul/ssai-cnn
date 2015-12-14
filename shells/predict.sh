#! /bin/bash

dname=MnihCNN_cis_2015-12-14_01-30-46
epoch=75
model=MnihCNN_cis
dataset=merged
channels=3

python scripts/predict.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--test_sat_dir data/mass_${dataset}/test/sat \
--channels ${channels} \
--offset 8
