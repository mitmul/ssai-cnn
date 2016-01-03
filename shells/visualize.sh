#! /bin/bash

export PYTHONPATH=".":$PYTHONPATH

model=MnihCNN_cis
epoch=400
function visualize() {
    python scripts/visualize.py \
    --model $1/$model.py \
    --param $1/epoch-$epoch.model \
    --out_dir $1/mid-$epoch
}

visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.88174
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.554376
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.609452
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.670797
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.770513
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_1.876234
visualize results/cis_1.0/MnihCNN_cis_2015-12-28_04-40-30_3.295603
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 110
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 120
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 130
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 140
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 150
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 160
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 170
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 180
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 190
# visualize MnihCNN_multi_2015-12-28_04-35-18 MnihCNN_multi 200
