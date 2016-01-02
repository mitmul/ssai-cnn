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

epoch=400
model=MnihCNN_cis
dataset=merged
channels=3

function predict_(){
    predict $1 $epoch $model $dataset $channels $2
}

# predict_ MnihCNN_multi_2015-12-31_21-46-34_2.106365 0
# predict_ MnihCNN_multi_2015-12-31_21-46-34_1.754149 2
# predict_ MnihCNN_multi_2015-12-31_21-46-34_1.751599 3
# predict_ MnihCNN_multi_2015-12-31_21-46-34_1.348473 4
# predict_ MnihCNN_multi_2015-12-31_21-46-34 5
# predict_ MnihCNN_multi_2015-12-31_21-46-33_1.89993 6
# predict_ MnihCNN_multi_2015-12-31_21-46-33 7
# predict_ MnihCNN_multi_2015-12-31_21-46-32 8

predict_ MnihCNN_cis_2015-12-31_21-47-24_2.3352 0
predict_ MnihCNN_cis_2015-12-31_21-47-24_1.880017 2
predict_ MnihCNN_cis_2015-12-31_21-47-24_1.784159 3
predict_ MnihCNN_cis_2015-12-31_21-47-24_1.519163 4
predict_ MnihCNN_cis_2015-12-31_21-47-24_1.242261 5
predict_ MnihCNN_cis_2015-12-31_21-47-24 6
predict_ MnihCNN_cis_2015-12-31_21-47-23_1.786241 7
predict_ MnihCNN_cis_2015-12-31_21-47-23 8

# predict_ MnihCNN_cis_2015-12-31_08-57-23 0
# predict_ MnihCNN_cis_2015-12-31_08-57-24 2
# predict_ MnihCNN_cis_2015-12-31_08-57-24_1.667062 3
# predict_ MnihCNN_cis_2015-12-31_08-57-24_1.823047 4
# predict_ MnihCNN_cis_2015-12-31_08-57-24_1.889858 5
# predict_ MnihCNN_cis_2015-12-31_08-57-24_2.430019 6
# predict_ MnihCNN_cis_2015-12-31_08-57-24_2.636972 7
# predict_ MnihCNN_cis_2015-12-31_08-57-24_3.248559 8

# predict_ MnihCNN_multi_2015-12-31_09-05-53 0
# predict_ MnihCNN_multi_2015-12-31_09-05-55 2
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.401205 3
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.423343 4
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.483466 5
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.623926 6
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.725186 7
# predict_ MnihCNN_multi_2015-12-31_09-05-55_1.885729 8

# predict_ MnihCNN_multi_2015-12-28_04-35-18 0
# predict_ MnihCNN_multi_2015-12-28_04-35-20_44 2
# predict_ MnihCNN_multi_2015-12-28_04-35-19 3
# predict_ MnihCNN_multi_2015-12-28_04-35-20_46 4
# predict_ MnihCNN_multi_2015-12-28_04-35-19_40 5
# predict_ MnihCNN_multi_2015-12-28_04-35-20_47 6
# predict_ MnihCNN_multi_2015-12-28_04-35-20 7
# predict_ MnihCNN_multi_2015-12-28_04-35-20_99 8

# predict_ MnihCNN_cis_2015-12-28_04-40-30 0
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.554376 2
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.609452 3
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.670797 4
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.770513 5
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.876234 6
# predict_ MnihCNN_cis_2015-12-28_04-40-30_1.88174 7
# predict_ MnihCNN_cis_2015-12-28_04-40-30_3.295603 8
