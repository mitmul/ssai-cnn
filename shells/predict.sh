#! /bin/bash

function predict() {
    nohup python scripts/predict.py \
    --model results/$7/$1/$3.py \
    --param results/$7/$1/epoch-$2.model \
    --test_sat_dir data/mass_$4/test/sat \
    --channels $5 \
    --offset 8 \
    --gpu $6 &
}

epoch=200
model=MnihCNN_cis
dataset=merged
channels=3
dname=cis

#predict MnihCNN_multi_2015-12-21_10-45-53 $epoch $model $dataset $channels 0 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_10 $epoch $model $dataset $channels 2 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_40 $epoch $model $dataset $channels 3 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_44 $epoch $model $dataset $channels 4 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_46 $epoch $model $dataset $channels 5 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_47 $epoch $model $dataset $channels 6 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_67 $epoch $model $dataset $channels 7 $dname
#predict MnihCNN_multi_2015-12-21_10-45-53_99 $epoch $model $dataset $channels 8 $dname

predict MnihCNN_cis_2015-12-21_12-48-31 $epoch $model $dataset $channels 0 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_24 $epoch $model $dataset $channels 2 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_40 $epoch $model $dataset $channels 3 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_44 $epoch $model $dataset $channels 4 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_46 $epoch $model $dataset $channels 5 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_47 $epoch $model $dataset $channels 6 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_67 $epoch $model $dataset $channels 7 $dname
predict MnihCNN_cis_2015-12-21_12-48-31_99 $epoch $model $dataset $channels 8 $dname
