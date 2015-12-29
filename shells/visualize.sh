#! /bin/bash

function visualize() {
    python scripts/visualize.py \
    --model models/$2.py \
    --param results/$1/epoch-$3.model \
    --out_dir results/$1/mid-$3
}

visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 20
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 10
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 20
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 30
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 40
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 50
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 60
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 70
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 80
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 90
visualize MnihCNN_cis_2015-12-28_04-40-30 MnihCNN_cis 100
