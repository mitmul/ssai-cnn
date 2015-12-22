#! /bin/bash

function visualize() {
    python scripts/visualize.py \
    --model results/$1/$2.py \
    --param results/$1/epoch-$3.model \
    --out_dir results/$1/mid-$3
}

visualize MnihCNN_cis_2015-12-19_05-11-12 MnihCNN_cis 200
visualize MnihCNN_multi_2015-12-19_05-11-12 MnihCNN_multi 200
