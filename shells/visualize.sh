#! /bin/bash

# dname=MnihCNN_cis_2015-12-19_05-11-12
# epoch=200
# model=MnihCNN_cis
dname=MnihCNN_multi_2015-12-19_05-11-12
epoch=200
model=MnihCNN_multi

python scripts/visualize.py \
--model results/${dname}/${model}.py \
--param results/${dname}/epoch-${epoch}.model \
--out_dir results/${dname}/weights-${epoch}
