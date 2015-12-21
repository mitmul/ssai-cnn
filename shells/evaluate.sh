#! /bin/bash

dname=MnihCNN_cis_2015-12-19_05-11-12
epoch=200
dataset=merged
channels=3
offset=8
pad=24

python scripts/evaluate.py \
--map_dir data/mass_${dataset}/test/map \
--result_dir results/${dname}/prediction_${epoch} \
--channel ${channels} \
--offset ${offset} \
--pad ${pad}
