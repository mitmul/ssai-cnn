#! /bin/bash

function predict() {
    nohup python scripts/predict.py \
    --model results/$1/$3.py \
    --param results/$1/epoch-$2.model \
    --test_sat_dir data/mass_$4/test/sat \
    --channels $5 \
    --offset 8 \
    --gpu $6 &
}

epoch=350
model=MnihCNN_cis
dataset=merged
channels=3

function predict_(){
    predict $1 $epoch $model $dataset $channels $2
}

# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-18 0
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-20_44 2
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-19 3
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-20_46 4
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-19_40 5
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-20_47 6
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-20 7
# predict_ predict_ MnihCNN_multi_2015-12-28_04-35-20_99 8
predict_ MnihCNN_cis_2015-12-28_04-40-30 0
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.554376 2
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.609452 3
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.670797 4
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.770513 5
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.876234 6
predict_ MnihCNN_cis_2015-12-28_04-40-30_1.88174 7
predict_ MnihCNN_cis_2015-12-28_04-40-30_3.295603 8
