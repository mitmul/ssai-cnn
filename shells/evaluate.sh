#! /bin/bash

dname=MnihCNN_cis_2015-12-18_15-06-59
epoch=99
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
