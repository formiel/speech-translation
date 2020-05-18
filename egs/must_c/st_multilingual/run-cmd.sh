#!/bin/bash

tag=$1
stage=$2
stop_stage=$2
ngpu=$3

set -x

# Check number of arguments
if [ $# -ge 2 ]; then
    echo "Running script ..."
else
    echo "Configuration file must be provided!"
    exit 1
fi

# Get the target languages from configuration name
if [[ $tag == *"lang"* ]]; then
    pos=$(echo $tag | grep -aob lang | grep -oE '[0-9]+')
    tgt_lang=$(echo ${tag:$(( pos + 5))})
else
    tgt_lang=de_es_fr_it_nl_pt_ro_ru
fi
echo "Run on target language pairs: ${tgt_lang}"

# Configuations
train_config=./conf/tuning/$tag.yaml
preprocess_config=./conf/specaug.yaml
decode_config=$4
trans_model=$5
trans_set=$6

bash run-multi.sh  --ngpu ${ngpu} --stage $stage --stop-stage $stop_stage \
        --tag $tag \
        --tgt-lang $tgt_lang \
        --train-config ${train_config} \
        --preprocess-config ${preprocess_config} \
        --decode-config ${decode_config} \
        --trans-model ${trans_model} \
        --trans-set ${trans_set}