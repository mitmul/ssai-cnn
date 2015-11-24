#! /bin/bash

dname=MnihCNN_cis_2015-11-24_03-41-07
epoch=7
model=MnihCNN_cis
python scripts/predict.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--test_sat_dir data/mass_merged/test/sat \
--channels 3
