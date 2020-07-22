#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=          # start from -1 if you need to start from data download
stop_stage=
ngpu=           # number of gpus ("0" uses cpu, otherwise use gpu)
nj=             # number of parallel jobs for decoding
debugmode=4
dumpdir=dump    # directory to dump full features
expdir=exp      # directory to save experiment folders
tensorboard_dir=tensorboard
datadir=        # directory where multilingual data folders are saved
N=0             # number of minibatches to be used (mainly for debugging). 
                # "0" uses all minibatches.
verbose=1       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
mode=debug
do_delta=false  # feature configuration

# Path to raw MuST-C data
must_c=

# target language related
tgt_langs=
# you can choose from de, es, fr, it, nl, pt, ro, ru
# To train the multilingual model, segment languages with _ as follows:
# e.g., tgt_lang="de_es_fr"

# pre-training related
asr_model=
st_model=

# training related
preprocess_config=

# preprocessing related
src_case=lc.rm              # lc.rm: lowercase with punctuation removal
tgt_case=tc                 # tc: truecase
use_joint_src_tgt_dict=     # if true, use one dictionary for source and target

# bpemode (unigram or bpe)
bpemode=bpe
nbpe=     # for target dictionary or joint source and target dictionary
nbpe_src= # for source dictionary only

# decoding related
decode_config=
trans_model=    # set a model to be used for decoding e.g. 'model.acc.best'
trans_set=
max_iter_eval=  # get best model upto a specified iteration

# model average related (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, use models with best validation.
                             # if false, use last `n_average` models.

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Modify params after parsing from command line
# Multilingual related parameters
tgt_langs=$(echo "$tgt_langs" | tr '_' '\n' | sort | tr '\n' '_')
tgt_langs=$(echo ${tgt_langs::-1})
lang_pairs=""
lang_count=0
for lang in $(echo ${tgt_langs} | tr '_' ' '); do
    lang_pairs+="en-${lang},"
    lang_count=$((lang_count + 1))
done
lang_pairs=$(echo ${lang_pairs::-1})

# suffix for dictionaries
if (( $lang_count == 8 )); then
    suffix="lgs_all8"
else
    suffix="lgs_${tgt_langs}"
fi

# training configuration
train_config=./conf/training/${tag}.yaml
if [ ${use_joint_src_tgt_dict} ]; then
    dprefix="dict1"
else
    dprefix="dict2"
fi

echo "*** General parameters ***"
echo "| ngpu: ${ngpu}"
echo "| experiment name: ${tag}"
echo "| target language(s): ${tgt_langs}"
echo "| number of target languages: ${lang_count}"
echo "| language pairs: ${lang_pairs}"

echo "*** Training-related parameters ***"
echo "| nbpe: ${nbpe}"
echo "| dictionary prefix: ${dprefix}"
echo "| dictionary suffix: ${suffix}"
echo "| train_config: ${train_config}"
echo "| preprocess_config: ${preprocess_config}"
echo "| pre-trained weights for encoder: ${asr_model}"
echo "| pre-trained weights for decoder: ${st_model}"

echo "*** Decoding-related parameters ***"
echo "| max_iter_eval: ${max_iter_eval}"
echo "| decode_config: ${decode_config}"
echo "| trans_model: ${trans_model}"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train, dev, and translation sets
train_set=train_sp.en-${tgt_langs}
train_dev=dev.en-${tgt_langs}

if [[ -z ${trans_set} ]]; then
    trans_set=""
    num_trans_set=0
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        trans_set+="tst-COMMON.en-${lang}.${lang} "
        num_trans_set=$(( num_trans_set + 1 ))
    done
else
    num_trans_set=0
    for set in $trans_set; do
        num_trans_set=$(( num_trans_set + 1 ))
    done
fi
echo "| trans sets: ${trans_set}"
echo "| number of trans sets: ${num_trans_set}"


if [[ ${stage} -le -1 ]] && [[ ${stop_stage} -ge -1 ]]; then
    echo "***** stage -1: Data Download *****"
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        local/download_and_untar.sh ${must_c} ${lang}
    done
fi


if [[ ${stage} -le 0 ]] && [[ ${stop_stage} -ge 0 ]]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "***** stage 0: Data Preparation *****"
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        local/data_prep.sh ${must_c} ${lang}
    done
fi


if [[ ${stage} -le 1 ]] && [[ ${stop_stage} -ge 1 ]]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "***** stage 1: Feature Generation *****"

    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        fbankdir="fbank_${lang}"
        train_set_lang=train_sp.en-${lang}.${lang}
        train_dev_lang=dev.en-${lang}.${lang}
        feat_tr_dir=${dumpdir}/${train_set_lang}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        feat_dt_dir=${dumpdir}/${train_dev_lang}/delta${do_delta}; mkdir -p ${feat_dt_dir}
        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        for x in train.en-${lang} dev.en-${lang} tst-COMMON.en-${lang} tst-HE.en-${lang}; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
                data/${x} exp/make_fbank/${x} ${fbankdir}
        done

        # speed-perturbed
        utils/perturb_data_dir_speed.sh 0.9 data/train.en-${lang} data/temp1.${lang}
        utils/perturb_data_dir_speed.sh 1.0 data/train.en-${lang} data/temp2.${lang}
        utils/perturb_data_dir_speed.sh 1.1 data/train.en-${lang} data/temp3.${lang}
        utils/combine_data.sh --extra-files utt2uniq data/train_sp.en-${lang} \
            data/temp1.${lang} data/temp2.${lang} data/temp3.${lang}
        rm -r data/temp1.${lang} data/temp2.${lang} data/temp3.${lang}

        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/train_sp.en-${lang} exp/make_fbank/train_sp.en-${lang} ${fbankdir}

        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${lang}/utt2spk > data/train_sp.en-${lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.tc.${lang} >data/train_sp.en-${lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.${lang} >data/train_sp.en-${lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.rm.${lang} >data/train_sp.en-${lang}/text.lc.rm.${lang}
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${lang}/utt2spk > data/train_sp.en-${lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.tc.${lang} >>data/train_sp.en-${lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.${lang} >>data/train_sp.en-${lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.rm.${lang} >>data/train_sp.en-${lang}/text.lc.rm.${lang}
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${lang}/utt2spk > data/train_sp.en-${lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.tc.${lang} >>data/train_sp.en-${lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.${lang} >>data/train_sp.en-${lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${lang}/utt_map <data/train.en-${lang}/text.lc.rm.${lang} >>data/train_sp.en-${lang}/text.lc.rm.${lang}

        # Divide into source and target languages
        for x in train_sp.en-${lang} dev.en-${lang} tst-COMMON.en-${lang} tst-HE.en-${lang}; do
            local/divide_lang.sh ${x} ${lang}
        done

        for x in train_sp.en-${lang} dev.en-${lang}; do
            # remove utt having more than 3000 frames
            # remove utt having more than 400 characters
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp

            # Match the number of utterances between source and target languages
            # extract common lines
            cut -f 1 -d " " data/${x}.en.tmp/text > data/${x}.${lang}.tmp/reclist1
            cut -f 1 -d " " data/${x}.${lang}.tmp/text > data/${x}.${lang}.tmp/reclist2
            comm -12 data/${x}.${lang}.tmp/reclist1 data/${x}.${lang}.tmp/reclist2 > data/${x}.en.tmp/reclist

            for lg in ${lang} en; do
                reduce_data_dir.sh data/${x}.${lg}.tmp data/${x}.en.tmp/reclist data/${x}.${lg}
                utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lg}
            done
            rm -rf data/${x}.*.tmp
        done

        # compute global CMVN
        compute-cmvn-stats scp:data/${train_set_lang}/feats.scp data/${train_set_lang}/cmvn.ark

        # dump features for training
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_set_lang}/delta${do_delta}/storage \
            ${feat_tr_dir}/storage
        fi
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_dev_lang}/delta${do_delta}/storage \
            ${feat_dt_dir}/storage
        fi
        dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
            data/${train_set_lang}/feats.scp data/${train_set_lang}/cmvn.ark exp/dump_feats/${train_set_lang} ${feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${train_dev_lang}/feats.scp data/${train_set_lang}/cmvn.ark exp/dump_feats/${train_dev_lang} ${feat_dt_dir}
        for ttask in ${trans_set}; do
            feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
            dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
                data/${ttask}/feats.scp data/${train_set_lang}/cmvn.ark exp/dump_feats/trans/${ttask} \
                ${feat_trans_dir}
        done
    done
fi


# Paths to dictionary and bpe model 
dname=${train_set}_${bpemode}${nbpe}units_${tgt_case}_${suffix}
dict=data/lang_1spm/use_${dprefix}/${dname}
nlsyms=data/lang_1spm/use_${dprefix}/${train_set}_non_lang_syms_${tgt_case}_${suffix}
nlsyms_tmp=data/lang_1spm/use_${dprefix}/${train_set}_non_lang_syms_${tgt_case}_tmp_${suffix}
bpemodel=data/lang_1spm/use_${dprefix}/${train_set}_${bpemode}${nbpe}_${tgt_case}_${suffix}
if [[ ${stage} -le 2 ]] && [[ ${stop_stage} -ge 2 ]]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "***** stage 2: Dictionary and Json Data Preparation *****"
    mkdir -p data/lang_1spm/use_${dprefix}

    # Create joint dictionary for both source and target languages
    if [ ${use_joint_src_tgt_dict} ]; then
        echo "*** Create a JOINT dictionary for source and target languages ***"
        dict=${dict}.txt
        nlsyms=${nlsyms}.txt
        nlsyms_tmp=${nlsyms_tmp}.txt
        echo "| source and target dictionary: ${dict}"
        echo "| source and target bpe model: ${bpemodel}"

        echo "make a non-linguistic symbol list for all languages"
        if [ -f ${nlsyms_tmp} ]; then
            echo "remove existing non-lang files"
            rm ${nlsyms_tmp}
        fi
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp}
            grep sp1.0 data/train_sp.en-${lang}.en/text.${src_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp}
        done
        
        cat ${nlsyms_tmp} | sort | uniq > ${nlsyms}
        rm ${nlsyms_tmp}
        cat ${nlsyms}

        echo "*** Make a joint source and target dictionary ***"
        # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        special_symbols="<unk> 1"
        i=2
        all_langs=$(echo "${tgt_langs}_en" | tr '_' ' ')
        all_langs_sorted=$(echo ${all_langs[*]}| tr " " "\n" | sort -n | tr "\n" " ")
        echo "all langs sorted: ${all_langs_sorted}"
        for lang in $(echo "${all_langs_sorted}" | tr '_' ' '); do
            special_symbols+="; <2${lang}> ${i}"
            i=$((i + 1))
        done
        sed "s/; /\n/g" <<< ${special_symbols} > ${dict}
        echo "special symbols"
        cat ${dict}

        offset=$(wc -l < ${dict})
        echo "| offset= ${offset}"
        input_path=data/lang_1spm/use_${dprefix}/input_${dprefix}_en-${tgt_langs}.${tgt_case}_${suffix}.txt
        if [ -f ${input_path} ]; then
            echo "remove existing input text file"
            rm ${input_path}
        fi

        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
            grep sp1.0 data/train_sp.en-${lang}.en/text.${src_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
            # cat data/lang_1spm/use_${dprefix}/input_en-${lang}.${lang}.${tgt_case}.txt >> ${input_path}
        done
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=${input_path} --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel}.model --output_format=piece < ${input_path} | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        echo "| number of tokens in dictionary: $(wc -l ${dict})"

        echo "make json files"
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            train_set_lang=train_sp.en-${lang}.${lang}
            train_dev_lang=dev.en-${lang}.${lang}
            feat_tr_dir_lang=${dumpdir}/${train_set_lang}/delta${do_delta}
            feat_dt_dir_lang=${dumpdir}/${train_dev_lang}/delta${do_delta}
            jname=data_${dprefix}_en-${lang}_${bpemode}${nbpe}_${tgt_case}_${suffix}.json

            data2json.sh --nj 16 --feat ${feat_tr_dir_lang}/feats.scp --text data/${train_set_lang}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${lang} \
                data/${train_set_lang} ${dict} > ${feat_tr_dir_lang}/${jname}
            data2json.sh --feat ${feat_dt_dir_lang}/feats.scp --text data/${train_dev_lang}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${lang} \
                data/${train_dev_lang} ${dict} > ${feat_dt_dir_lang}/${jname}
            
            trans_set_lang="tst-COMMON.en-${lang}.${lang} tst-HE.en-${lang}.${lang}"
            for ttask in ${trans_set_lang}; do
                feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
                data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${ttask}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${lang} \
                    data/${ttask} ${dict} > ${feat_trans_dir}/${jname}
            done

            # update json (add source references)
            sets="${train_set_lang} ${train_dev_lang}"
            sets+=" "${trans_set_lang}
            for x in ${sets}; do
                echo "add source references to ${x}"
                feat_dir=${dumpdir}/${x}/delta${do_delta}
                data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${lang}.en
                update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                    ${feat_dir}/${jname} ${data_dir} ${dict}
            done
        done

    # Create separate dictionaries: 1 for source transcription, 1 joint for all target langs
    else 
        echo "*** Create SEPARATE dictionaries for source and target languages ***"
        dict_src=${dict}.src.txt
        nlsyms_src=${nlsyms}.src.txt
        nlsyms_tmp_src=${nlsyms_tmp}.src.txt
        bpemodel_src=${bpemodel}.src

        dict_tgt=${dict}.tgt.txt
        nlsyms_tgt=${nlsyms}.tgt.txt
        nlsyms_tmp_tgt=${nlsyms_tmp}.tgt.txt
        bpemodel_tgt=${bpemodel}.tgt

        echo "| source dictionary: ${dict_src}"
        echo "| target dictionary: ${dict_tgt}"
        echo "| source bpe model: ${bpemodel_src}"
        echo "| target bpe model: ${bpemodel_tgt}"

        # Create joint dictionary for target languages
        echo "make a non-linguistic symbol list for target languages"
        if [ -f ${nlsyms_tmp_tgt} ]; then
            echo "remove existing non-lang files"
            rm ${nlsyms_tmp_tgt}
        fi
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp_tgt}
        done
        
        echo "Target non-lang symbols"
        cat ${nlsyms_tmp_tgt} | sort | uniq > ${nlsyms_tgt}
        rm ${nlsyms_tmp_tgt}
        cat ${nlsyms_tgt}

        echo "*** Make a joint target dictionary ***"
        # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        # add more special symbols for later use
        # special_symbols="<unk> 1; <s> 2; </s> 3; <pad> 4; <TRANS> 5; <RECOG> 6; <DELAY> 7"
        special_symbols="<unk> 1"
        i=2
        tgt_langs_sorted=$(echo ${tgt_langs[*]}| tr " " "\n" | sort -n | tr "\n" " ")
        echo "target langs sorted: ${tgt_langs_sorted}"
        for lang in $(echo "${tgt_langs_sorted}" | tr '_' ' '); do
            special_symbols+="; <2${lang}> ${i}"
            i=$((i + 1))
        done
        sed "s/; /\n/g" <<< ${special_symbols} > ${dict_tgt}
        echo "special symbols"
        cat ${dict_tgt}

        offset=$(wc -l < ${dict_tgt})
        echo "| offset= ${offset}"
        input_path=data/lang_1spm/use_${dprefix}/input_${dprefix}_tgt_${tgt_langs}.${tgt_case}_${suffix}.txt
        if [ -f ${input_path} ]; then
            echo "remove existing input text file"
            rm ${input_path}
        fi

        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
        done
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms_tgt})" --input=${input_path} --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel_tgt} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel_tgt}.model --output_format=piece < ${input_path} | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_tgt}
        echo "| number of tokens in dictionary: $(wc -l ${dict_tgt})"

        echo "make json files"
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            train_set_lang=train_sp.en-${lang}.${lang}
            train_dev_lang=dev.en-${lang}.${lang}
            feat_tr_dir_lang=${dumpdir}/use_${dprefix}/${train_set_lang}/delta${do_delta}
            feat_dt_dir_lang=${dumpdir}/use_${dprefix}/${train_dev_lang}/delta${do_delta}
            jname=data_${dprefix}_en-${lang}_${bpemode}${nbpe}.${tgt_case}_${suffix}.json

            data2json.sh --nj 16 --feat ${feat_tr_dir_lang}/feats.scp --text data/${train_set_lang}/text.${tgt_case} --bpecode ${bpemodel_tgt}.model --lang ${lang} \
                data/${train_set_lang} ${dict_tgt} > ${feat_tr_dir_lang}/${jname}
            data2json.sh --feat ${feat_dt_dir_lang}/feats.scp --text data/${train_dev_lang}/text.${tgt_case} --bpecode ${bpemodel_tgt}.model --lang ${lang} \
                data/${train_dev_lang} ${dict_tgt} > ${feat_dt_dir_lang}/${jname}
            
            trans_set_lang="tst-COMMON.en-${lang}.${lang} tst-HE.en-${lang}.${lang}"
            for ttask in ${trans_set_lang}; do
                feat_trans_dir=${dumpdir}/use_${dprefix}/${ttask}/delta${do_delta}
                data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${ttask}/text.${tgt_case} --bpecode ${bpemodel_tgt}.model --lang ${lang} \
                    data/${ttask} ${dict_tgt} > ${feat_trans_dir}/${jname}
            done
        done

        # Create dictionay for source
        # Create joint dictionary for target languages
        echo "make a non-linguistic symbol list for source language"
        if [ -f ${nlsyms_tmp_src} ]; then
            echo "remove existing non-lang files"
            rm ${nlsyms_tmp_src}
        fi
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.en/text.${src_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp_src}
        done
        
        echo "Source non-lang symbols"
        cat ${nlsyms_tmp_src} | sort | uniq > ${nlsyms_src}
        rm ${nlsyms_tmp_src}
        cat ${nlsyms_src}

        echo "*** Make a source dictionary ***"
        # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        special_symbols="<unk> 1; <2en> 2"
        sed "s/; /\n/g" <<< ${special_symbols} > ${dict_src}
        echo "special symbols"
        cat ${dict_src}

        offset=$(wc -l < ${dict_src})
        echo "| offset= ${offset}"
        input_path=data/lang_1spm/use_${dprefix}/input_${dprefix}_src-en_${tgt_langs}.${src_case}_${suffix}.txt
        if [ -f ${input_path} ]; then
            echo "remove existing input text file"
            rm ${input_path}
        fi

        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.en-${lang}.en/text.${src_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
        done
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms_src})" --input=${input_path} --vocab_size=${nbpe_src} --model_type=${bpemode} --model_prefix=${bpemodel_src} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel_src}.model --output_format=piece < ${input_path} | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_src}
        echo "| number of tokens in source dictionary: $(wc -l ${dict_src})"

        echo "make json files"
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            train_set_lang=train_sp.en-${lang}.${lang}
            train_dev_lang=dev.en-${lang}.${lang}
            trans_set_lang="tst-COMMON.en-${lang}.${lang} tst-HE.en-${lang}.${lang}"

            sets="${train_set_lang} ${train_dev_lang}"
            sets+=" "${trans_set_lang}
            # update json (add source references)
            for x in ${sets}; do
                echo "add source references to ${x}"
                feat_dir=${dumpdir}/use_${dprefix}/${x}/delta${do_delta}
                data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${lang}.en
                update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel_src}.model \
                    ${feat_dir}/${jname} ${data_dir} ${dict_src}
            done
        done
    fi
fi

# NOTE: skip stage 3: LM Preparation

# Experiment name and data directory
if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${st_model}" ]; then
        expname=${expname}_sttrans
    fi
else
    expname=${tag} # use tag for experiment name
fi

# Experiment and tensorboard directories
expdir=${expdir}/${expname}
tensorboard_dir=${tensorboard_dir}/${expname}
mkdir -p ${expdir}
echo "| expdir: ${expdir}"
echo "| tensorboard_dir: ${tensorboard_dir}"

# Data input folders
datadir=${datadir}/${tgt_langs}
train_json_dir=${datadir}/use_${dprefix}/train_sp
val_json_dir=${datadir}/use_${dprefix}/dev
dict=${datadir}/lang_1spm/use_${dprefix}/${dname}.txt

echo "*** Paths to training data and dictionary ***"
echo "| train_json_dir: ${train_json_dir}"
echo "| val_json_dir: ${val_json_dir}"
echo "| dictionary: ${dict}"

if [[ ${stage} -le 4 ]] && [[ ${stop_stage} -ge 4 ]]; then
    echo "***** stage 4: Network Training *****"

    # Find the last snapshot to resume training
    resume_dir=$expdir/results
    exist_snaphots=false
    for i in $resume_dir/snapshot.iter.*; do test -f "$i" && exist_snaphots=true && break; done
    if [ "${exist_snaphots}" = true ]; then
        ckpt_nums=$(ls $resume_dir | grep snapshot | sed 's/[^0-9]*//g' | sed 's/\n/" "/g')
        last_ep=$(echo "${ckpt_nums[*]}" | sort -nr | head -n1)
        echo "Resume training from the last snapshot ${resume_dir}/snapshot.iter.${last_ep}"
        resume=${resume_dir}/snapshot.iter.${last_ep}
    fi

    # Run training
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --lang-pairs ${lang_pairs} \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir ${tensorboard_dir} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume $resume \
        --train-json ${train_json_dir} \
        --valid-json ${val_json_dir} \
        --enc-init ${asr_model} \
        --dec-init ${st_model}
    
    echo "Log output is saved in ${expdir}/train.log"
fi


if [[ ${stage} -le 5 ]] && [[ ${stop_stage} -ge 5 ]]; then
    echo "***** stage 5: Decoding *****"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        if [[ -z ${trans_model} ]]; then
            if [ ${max_iter_eval} > 0 -a ${max_iter_eval} -le ${last_ep} ]; then
                # Get the model with best validation accuray up until that iter
                trans_model=model.${max_iter_eval}.acc.best
                opt="--log ${expdir}/results/log"
            elif ${use_valbest_average}; then
                trans_model=model.val${n_average}.avg.best
                opt="--log ${expdir}/results/log"
            else
                trans_model=model.last${n_average}.avg.best
                opt="--log"
            fi
            echo "| trans_model: ${trans_model}"

            if [[ ! -f ${expdir}/results/${trans_model} ]]; then
                echo "*** Get trans_model ***"
                local/average_checkpoints_st.py \
                    ${opt} \
                    --max-iter-eval ${max_iter_eval} \
                    --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.iter.* \
                    --out ${expdir}/results/${trans_model} \
                    --num ${n_average}        
            else
                echo "trans_model existed: ${expdir}/results/${trans_model}"
            fi
        fi
    fi

    # Use all threads available
    nj=`grep -c ^processor /proc/cpuinfo`
    nj=$(( nj / num_trans_set ))

    if [[ $tag == *"debug"* ]]; then
        nj=1 # for debug
    fi
    echo "| njobs = ${nj}"
    pids=() # initialize pids

    for ttask in ${trans_set}; do
        split=$(cut -d'.' -f1 <<< ${ttask})
        lg_pair=$(cut -d'.' -f2 <<< ${ttask})
        lg_tgt=$(cut -d'.' -f3 <<< ${ttask})
        echo "| split: ${split}"
        echo "| language pair: ${lg_pair}"
        echo "| target language: ${lg_tgt}"
    
    (
        decode_config_lg_pair=${decode_config}.${lg_pair}.yaml
        decode_dir=decode_$(basename ${train_config%.*})_$(basename ${decode_config})_${split}_${lg_pair}_${trans_model}
        feat_trans_dir=${datadir}/${split}
        echo "| decode_dir: ${decode_dir}"
        echo "| feat_trans_dir: ${feat_trans_dir}"

        if [ ! -f "${feat_trans_dir}/split${nj}utt_${tgt_langs}/${lg_pair}.${nj}.json" ]; then
            # split data
            splitjson.py --parts ${nj} --tgt_lang ${tgt_langs} ${feat_trans_dir}/${lg_pair}.json
            echo "Finished splitting json file."
        else
            echo "json file has been already split."
        fi

        #### use CPU for decoding
        ngpu=0

        if [[ ! -f "${expdir}/${decode_dir}/data.json" ]]; then
            echo "Start decoding..."
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                st_trans.py \
                --config ${decode_config_lg_pair} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --trans-json ${feat_trans_dir}/split${nj}utt_${tgt_langs}/${lg_pair}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${trans_model} \
                --verbose ${verbose}
        fi

        # Compute BLEU
        if [[ ! -s "${expdir}/${decode_dir}/result.tc.txt" ]]; then
            echo "Compute BLEU..."
            chmod +x local/score_bleu_st.sh
            local/score_bleu_st.sh --case ${tgt_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                ${expdir}/${decode_dir} ${lg_tgt} ${dict}
        else
            echo "BLEU has been computed."
            cat ${expdir}/${decode_dir}/result.tc.txt
        fi

        # Compute WER
        if [[ ! -s "${expdir}/${decode_dir}/result.wrd.wer.txt" ]]; then
            echo "Compute WER score..."
            chmod +x local/score_sclite_st.sh
            local/score_sclite_st.sh --case ${src_case} --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
                ${expdir}/${decode_dir} ${dict}
        else
            echo "WER has been computed."
            cat ${expdir}/${decode_dir}/result.wrd.wer.txt
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi