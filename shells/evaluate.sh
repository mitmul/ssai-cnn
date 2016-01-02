#! /bin/bash

epoch=250
dataset=merged
channels=3
offset=8
pad=24
relax=0
steps=512

function evaluate() {
    nohup python scripts/evaluate.py \
    --map_dir data/mass_$dataset/test/map \
    --result_dir $1/prediction_$epoch \
    --channel $channels \
    --offset $offset \
    --pad $pad \
    --relax $relax \
    --steps $steps &
}

evaluate results/MnihCNN_multi_2015-12-21_10-45-53
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_10
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_40
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_44
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_46
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_47
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_67
evaluate results/MnihCNN_multi_2015-12-21_10-45-53_99

# evaluate MnihCNN_cis_2015-12-21_12-48-31 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_24 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_40 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_44 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_46 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_47 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_67 $epoch $dataset $channels $offset $pad
# evaluate MnihCNN_cis_2015-12-21_12-48-31_99 $epoch $dataset $channels $offset $pad
