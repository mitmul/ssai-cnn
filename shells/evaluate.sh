#! /bin/bash

function evaluate() {
    nohup python scripts/evaluate.py \
    --map_dir data/mass_$3/test/map \
    --result_dir results/$1/prediction_$2 \
    --channel $4 \
    --offset $5 \
    --pad $6 \
    --relax $7 &
}

epoch=250
dataset=merged
channels=3
offset=8
pad=24
relax=0

evaluate MnihCNN_multi_2015-12-21_10-45-53 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_10 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_40 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_44 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_46 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_47 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_67 $epoch $dataset $channels $offset $pad $relax
evaluate MnihCNN_multi_2015-12-21_10-45-53_99 $epoch $dataset $channels $offset $pad $relax

# evaluate MnihCNN_cis_2015-12-21_12-48-31 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_24 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_40 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_44 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_46 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_47 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_67 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_99 $epoch $dataset $channels $offset $pad
