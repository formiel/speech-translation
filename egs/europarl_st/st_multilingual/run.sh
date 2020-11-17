#!/bin/bash

# Modified by Hang Le
# The original copyright is appended below
# --
# Copyright 2019 Kyoto University (Hirofumi Inaguma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch     # chainer or pytorch
stage=              # start from -1 if you need to start from data download
stop_stage=
ngpu=               # number of gpus ("0" uses cpu, otherwise use gpu)
nj=48               # number of parallel jobs for decoding
debugmode=4
dumpdir=dump        # directory to dump full features
expdir=exp          # directory to save experiment folders
tensorboard_dir=tensorboard
datadir=            # directory where multilingual data folders are saved
train_config_dir=   # directory where training configs are saved
decode_config_dir=  # directory where decode confis are saved
N=0                 # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1           # verbose option
resume=             # Resume the training from snapshot
seed=1              # seed to generate random number
do_delta=false      # feature configuration

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
europarl_st=

# source and target languages
# you can choose from de, en, es, fr, it, nl, pl, pt, ro
# if you want to train many-to-many model, segment source languages with _ 
# as follows: e.g., tgt_lang="de_en_es"
all_langs=de_en_es_fr_it_nl_pl_pt_ro
src_lang=en
tgt_langs=

# pre-training related
asr_model=
st_model=
init_from_decoder_asr=
init_from_decoder_mt=

# training related
preprocess_config=
do_st=                       # if false, train ASR model
use_adapters=                # if true, use adapter for fine-tuning
train_adapters=false         # if true, train adapter from scratch
use_adapters_for_asr=        # if true, then add adapters for transcription
use_adapters_in_enc=         # if true, use adapters in encoder
do_mt=false                  # if true then train MT model
early_stop_criterion=validation/main/acc
dict=                        
bpemodel=

# preprocessing related
src_case=lc.rm       # lc.rm: lowercase with punctuation removal
tgt_case=tc          # tc: truecase
use_lid=true         # if false then not use language id (for bilingual systems)
use_joint_dict=true  # if true, use one dictionary for source and target
use_multi_dict=false # if true, use dictionary for all target languages

# bpemode (unigram or bpe)
bpemode=bpe
nbpe=             # for target dictionary or joint source and target dictionary
nbpe_src=         # for source dictionary only

# decoding related
decode_config=    # configuration for decoding
trans_model=      # set a model to be used for decoding e.g. 'model.acc.best'
trans_set=        # data set to decode
max_iter_eval=    # get best model up to this iteration
min_iter_eval=    # get best model from this iteration
remove_non_verbal_eval=true  # if true, then remove non-verbal tokens in evaluation

# model average related (only for transformer)
n_average=1                  # the number of ST models to be averaged,
                             # 1: disable model averaging and choose best model.
use_valbest_average=true     # if true, use models with best validation.
                             # if false, use last `n_average` models.

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# training configuration
train_config=${train_config_dir}/${tag}.yaml

# Language related parameters
if [[ -z ${tgt_langs} ]]; then
    if [[ ${src_lang} == "ro" ]]; then
        tgt_langs=$(echo "${all_langs//_${src_lang}/}")
    else
        tgt_langs=$(echo "${all_langs//${src_lang}_/}")
    fi
fi

tgt_langs=$(echo "$tgt_langs" | tr '_' '\n' | sort | tr '\n' '_')
tgt_langs=$(echo ${tgt_langs::-1})
lang_pairs=""
lang_count=0
for lang in $(echo ${tgt_langs} | tr '_' ' '); do
    lang_pairs+="${src_lang}-${lang},"
    lang_count=$((lang_count + 1))
done
lang_pairs=$(echo ${lang_pairs::-1})

# use language ID if there is more than 1 target languages
if (( $lang_count != 1 )); then
    use_lid=true
fi

if (( $lang_count == 7 )) || [[ ${use_adapters} == "true" ]] || [[ ${tag} == *"full_ft"* ]] || [[ ${tag} == *"trained_on_mustc"* ]]; then
    prefix_tmp="lgs_all"
else
    prefix_tmp="lgs_${tgt_langs}"
fi

dprefix="dict1" # prefix for dictionaries
suffix_mt=
# suffix for dictionaries
if [[ ${use_adapters} == "true" ]]; then
    train_type="adapters"
else
    if [[ ${tag} == *"full_ft"* ]] || [[ ${tag} == *"trained_on_mustc"* ]]; then
        train_type="full_ft"
    else
        train_type="pretrain"
    fi
fi
suffix="${prefix_tmp}_${train_type}"

# if [[ ${suffix} == *"adapters"* ]] || [[ ${suffix} == *"full_ft"* ]]; then
#     if [ -z ${dict} ] || [ -z ${bpemodel} ]; then
#         echo "dict and bpemodel are required for adapter-based fine_tuning."
#         exit 1
#     fi
# fi

echo "*** General parameters ***"
echo "| ngpu: ${ngpu}"
echo "| experiment name: ${tag}"
echo "| target language(s): ${tgt_langs}"
echo "| number of target languages: ${lang_count}"
echo "| language pairs: ${lang_pairs}"

echo "*** Training-related parameters ***"
echo "| nbpe: ${nbpe}"
echo "| nbpe_src: ${nbpe_src}"
echo "| dictionary prefix: ${dprefix}"
echo "| dictionary suffix: ${suffix}"
echo "| dict: ${dict}"
echo "| bpemodel: ${bpemodel}"
echo "| use language ID: ${use_lid}"
echo "| use adapters: ${use_adapters}"
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

train_set=train_sp.${src_lang}-${tgt_langs}
train_dev=dev.${src_lang}-${tgt_langs}

train_set_dict=${train_set}
if [[ ${train_adapters} == "true" ]] || [[ ${use_multi_dict} == "true" ]]; then
    train_set_dict=train_sp.${src_lang}-de_es_fr_it_nl_pt_ro
fi

num_trans_set=0
if [[ -z ${trans_set} ]]; then
    trans_set=""
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        trans_set="${trans_set} test.${src_lang}-${lang}"
        num_trans_set=$(( num_trans_set + 1 ))
    done
else
    for set in ${trans_set}; do
        num_trans_set=$(( num_trans_set + 1 ))
    done
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${europarl_st}
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        local/data_prep.sh ${europarl_st} ${src_lang} ${lang}
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        train_set_lg=train_sp.${src_lang}-${lang}.${lang}
        train_dev_lg=dev.${src_lang}-${lang}.${lang}
        trans_set_lg=test.${src_lang}-${lang}.${lang}
        feat_tr_dir_lg=${dumpdir}/${train_set_lg}/delta${do_delta}; mkdir -p ${feat_tr_dir_lg}
        feat_dt_dir_lg=${dumpdir}/${train_dev_lg}/delta${do_delta}; mkdir -p ${feat_dt_dir_lg}
        feat_trans_dir=${dumpdir}/${trans_set_lg}/delta${do_delta}; mkdir -p ${feat_trans_dir}
        for x in train.${src_lang}-${lang} dev.${src_lang}-${lang} test.${src_lang}-${lang}; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
                data/${x} exp/make_fbank/${x} ${fbankdir}
        done
        
        echo "Run speed pertubation..."
        # speed-perturbed
        utils/perturb_data_dir_speed.sh 0.9 data/train.${src_lang}-${lang} data/temp1.${lang}
        utils/perturb_data_dir_speed.sh 1.0 data/train.${src_lang}-${lang} data/temp2.${lang}
        utils/perturb_data_dir_speed.sh 1.1 data/train.${src_lang}-${lang} data/temp3.${lang}
        utils/combine_data.sh --extra-files utt2uniq data/train_sp.${src_lang}-${lang} \
            data/temp1.${lang} data/temp2.${lang} data/temp3.${lang}
        rm -r data/temp1.${lang} data/temp2.${lang} data/temp3.${lang}
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/train_sp.${src_lang}-${lang} exp/make_fbank/train_sp.${src_lang}-${lang} ${fbankdir}

        for lg in ${src_lang} ${lang}; do
            echo "Create utt_map..."
            awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.${src_lang}-${lang}/utt2spk > data/train_sp.${src_lang}-${lang}/utt_map
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.tc.${lg} >data/train_sp.${src_lang}-${lang}/text.tc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.${lg} >data/train_sp.${src_lang}-${lang}/text.lc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.rm.${lg} >data/train_sp.${src_lang}-${lang}/text.lc.rm.${lg}
            awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.${src_lang}-${lang}/utt2spk > data/train_sp.${src_lang}-${lang}/utt_map
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.tc.${lg} >>data/train_sp.${src_lang}-${lang}/text.tc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.${lg} >>data/train_sp.${src_lang}-${lang}/text.lc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.rm.${lg} >>data/train_sp.${src_lang}-${lang}/text.lc.rm.${lg}
            awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.${src_lang}-${lang}/utt2spk > data/train_sp.${src_lang}-${lang}/utt_map
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.tc.${lg} >>data/train_sp.${src_lang}-${lang}/text.tc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.${lg} >>data/train_sp.${src_lang}-${lang}/text.lc.${lg}
            utils/apply_map.pl -f 1 data/train_sp.${src_lang}-${lang}/utt_map <data/train.${src_lang}-${lang}/text.lc.rm.${lg} >>data/train_sp.${src_lang}-${lang}/text.lc.rm.${lg}
        done

        # Divide into source and target languages
        for x in train_sp.${src_lang}-${lang} dev.${src_lang}-${lang} test.${src_lang}-${lang}; do
            local/divide_lang.sh ${x} ${src_lang} ${lang}
        done

        for x in train_sp.${src_lang}-${lang} dev.${src_lang}-${lang}; do
            # remove utt having more than 3000 frames
            # remove utt having more than 400 characters
            for lg in ${lang} ${src_lang}; do
                remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lg} data/${x}.${lg}.tmp
            done

            # Match the number of utterances between source and target languages
            # extract commocn lines
            cut -f 1 -d " " data/${x}.${src_lang}.tmp/text > data/${x}.${lang}.tmp/reclist1
            cut -f 1 -d " " data/${x}.${lang}.tmp/text > data/${x}.${lang}.tmp/reclist2
            comm -12 data/${x}.${lang}.tmp/reclist1 data/${x}.${lang}.tmp/reclist2 > data/${x}.${src_lang}.tmp/reclist

            for lg in ${lang} ${src_lang}; do
                reduce_data_dir.sh data/${x}.${lg}.tmp data/${x}.${src_lang}.tmp/reclist data/${x}.${lg}
                utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lg}
            done
            rm -r data/${x}.*.tmp
        done

        # compute global CMVN
        echo "compute global CMVN..."     
        compute-cmvn-stats scp:data/${train_set_lg}/feats.scp data/${train_set_lg}/cmvn.ark

        # dump features for training
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir_lg}/storage ]; then
            echo "dump features for train set..."
            utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_set_lg}/delta${do_delta}/storage \
            ${feat_tr_dir_lg}/storage
        fi
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir_lg}/storage ]; then
            echo "dump features for dev set..."
            utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_dev_lg}/delta${do_delta}/storage \
            ${feat_dt_dir_lg}/storage
        fi

        echo "dumping features ..."
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
            data/${train_set_lg}/feats.scp data/${train_set_lg}/cmvn.ark exp/dump_feats/${train_set_lg} ${feat_tr_dir_lg}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
            data/${train_dev_lg}/feats.scp data/${train_set_lg}/cmvn.ark exp/dump_feats/${train_dev_lg} ${feat_dt_dir_lg}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
            data/${trans_set_lg}/feats.scp data/${train_set_lg}/cmvn.ark exp/dump_feats/trans/${trans_set_lg} ${feat_trans_dir}
    done
    echo "Successully proceed features."
fi

if [[ ${suffix} == *"adapters"* ]] || [[ ${suffix} == *"full_ft"* ]]; then
    # Use dictionary of MuST-C
    main_dir=/gpfswork/rech/dbn/umz16dj
    mustc_dir=espnet/egs/must_c/st_multilingual
    mustc_lgs=de_es_fr_it_nl_pt_ro_ru
    dict=${main_dir}/Data/after_submisson/mustc_espnet/${mustc_lgs}/use_dict1/src8000_tgt8000/lang_1spm/train_sp.en-${mustc_lgs}_bpe8000_tc_lgs_all8.txt
    nlsyms=${main_dir}/${mustc_dir}/data/lang_1spm/use_dict1/train_sp.en-${mustc_lgs}_non_lang_syms_tc_lgs_all8.txt
    bpemodel=$main_dir/${mustc_dir}/data/lang_1spm/use_dict1/train_sp.en-${mustc_lgs}_bpe8000_tc_lgs_all8
else
    dname=${train_set}_${bpemode}${nbpe}_${tgt_case}_${suffix}${suffix_mt}
    bpemodel=data/lang_1spm/use_${dprefix}/${dname}
    dict=${bpemodel}.txt
    nlsyms=data/lang_1spm/use_${dprefix}/${train_set}_non_lang_syms_${tgt_case}_${suffix}${suffix_mt}.txt
    nlsyms_tmp=data/lang_1spm/use_${dprefix}/${train_set}_non_lang_syms_tmp_${tgt_case}_${suffix}${suffix_mt}.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/use_${dprefix}
    echo "| dict: ${dict}"
    echo "| bpemodel: ${bpemodel}"

    if [[ ${suffix} == *"pretrain"* ]]; then
        echo "make a non-linguistic symbol list for all languages"
        if [ -f ${nlsyms_tmp} ]; then
            echo "remove existing non-lang files"
            rm ${nlsyms_tmp}
        fi
        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.${src_lang}-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp}
            grep sp1.0 data/train_sp.${src_lang}-${lang}.${src_lang}/text.${src_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;' >> ${nlsyms_tmp}
        done
        
        cat ${nlsyms_tmp} | sort | uniq > ${nlsyms}
        rm ${nlsyms_tmp}
        cat ${nlsyms}

        echo "*** Make a joint source and target dictionary ***"
        # echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
        special_symbols="<unk> 1"
        if [[ ${use_lid} == "true" ]]; then
            i=2
            all_langs=$(echo "${tgt_langs}_${src_lang}" | tr '_' ' ')
            all_langs_sorted=$(echo ${all_langs[*]}| tr " " "\n" | sort -n | tr "\n" " ")
            echo "all langs sorted: ${all_langs_sorted}"
            for lang in $(echo "${all_langs_sorted}" | tr '_' ' '); do
                special_symbols+="; <2${lang}> ${i}"
                i=$((i + 1))
            done
        fi
        sed "s/; /\n/g" <<< ${special_symbols} > ${dict}
        echo "special symbols"
        cat ${dict}

        offset=$(wc -l < ${dict})
        echo "| offset= ${offset}"
        input_path=data/lang_1spm/use_${dprefix}/input_${dprefix}_${bpemode}${nbpe}_${src_lang}-${tgt_langs}_${tgt_case}_${suffix}${suffix_mt}.txt
        if [ -f ${input_path} ]; then
            echo "remove existing input text file"
            rm ${input_path}
        fi

        for lang in $(echo ${tgt_langs} | tr '_' ' '); do
            grep sp1.0 data/train_sp.${src_lang}-${lang}.${lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
            grep sp1.0 data/train_sp.${src_lang}-${lang}.${src_lang}/text.${src_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> ${input_path}
        done
        spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=${input_path} --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${bpemodel}.model --output_format=piece < ${input_path} | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
        echo "| number of tokens in dictionary: $(wc -l ${dict})"
    fi

    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        echo "make json files"
        train_set_lg=train_sp.${src_lang}-${lang}.${lang}
        train_dev_lg=dev.${src_lang}-${lang}.${lang}
        trans_set_lg=test.${src_lang}-${lang}.${lang}
        feat_tr_dir_lg=${dumpdir}/${train_set_lg}/delta${do_delta}
        feat_dt_dir_lg=${dumpdir}/${train_dev_lg}/delta${do_delta}
        feat_trans_dir=${dumpdir}/${trans_set_lg}/delta${do_delta}
        jname=data_${dprefix}_${src_lang}-${lang}_${bpemode}${nbpe}_${tgt_case}_${suffix}${suffix_mt}.json

        data2json.sh --nj ${nj} --feat ${feat_tr_dir_lg}/feats.scp \
            --text data/${train_set_lg}/text.${tgt_case} \
            --bpecode ${bpemodel}.model --lang ${lang} \
            data/${train_set_lg} ${dict} > ${feat_tr_dir_lg}/${jname}
        data2json.sh --feat ${feat_dt_dir_lg}/feats.scp \
            --text data/${train_dev_lg}/text.${tgt_case} \
            --bpecode ${bpemodel}.model --lang ${lang} \
            data/${train_dev_lg} ${dict} > ${feat_dt_dir_lg}/${jname}
        data2json.sh --feat ${feat_trans_dir}/feats.scp \
            --text data/${trans_set_lg}/text.${tgt_case} \
            --bpecode ${bpemodel}.model --lang ${lang} \
            data/${trans_set_lg} ${dict} > ${feat_trans_dir}/${jname}

        # update json (add source references)
        for x in ${train_set_lg} ${train_dev_lg} ${trans_set_lg}; do
            feat_dir=${dumpdir}/${x}/delta${do_delta}
            data_dir=data/$(echo ${x} | cut -f 1 -d ".").${src_lang}-${lang}.${src_lang}
            update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
                ${feat_dir}/${jname} ${data_dir} ${dict}
        done
    done
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

# Data folders
datadir_tmp=${datadir}
datadir=${datadir_tmp}/${tgt_langs}/use_${dprefix}/src${nbpe_src}_tgt${nbpe}/${train_type}
if [[ ${use_multi_dict} == "true" ]]; then
    datadir=${datadir_tmp}/de_es_fr_it_nl_pt_ro/use_${dprefix}/src${nbpe_src}_tgt${nbpe}/${train_type}
# elif (( $lang_count == 1 )) && [[ ${train_adapters} == "true" ]]; then
#     datadir=${datadir_tmp}/${tgt_langs}_train_adapters/use_${dprefix}/src${nbpe_src}_tgt${nbpe}
# elif (( $lang_count == 1 )) && [[ ${train_adapters} == "false" ]]; then
#     datadir=${datadir}/use_lid_${use_lid}
fi

if (( $lang_count == 1 )) && [[ ${train_adapters} == "false" ]]; then
    train_json_dir=${datadir}/train_sp/${src_lang}-${tgt_langs}.json
    val_json_dir=${datadir}/dev/${src_lang}-${tgt_langs}.json 
else
    train_json_dir=${datadir}/train_sp
    val_json_dir=${datadir}/dev
fi

if [[ ${use_joint_dict} == "true" ]]; then
    dname=${train_set_dict}_${bpemode}${nbpe}_${tgt_case}_${suffix}
    dict_tgt=${datadir}/lang_1spm/${dname}.txt
    dict_src=${datadir}/lang_1spm/${dname}.txt
    bpemodel_tgt=data/lang_1spm/use_${dprefix}/${dname}
    bpemodel_src=data/lang_1spm/use_${dprefix}/${dname}
else
    dname=${train_set_dict}_${bpemode}_src${nbpe_src}${src_case}_tgt${nbpe}${tgt_case}_${suffix}
    dict_tgt=${datadir}/lang_1spm/${dname}.tgt.txt
    dict_src=${datadir}/lang_1spm/${dname}.src.txt
    bpemodel_tgt=data/lang_1spm/use_${dprefix}/${dname}.tgt
    bpemodel_src=data/lang_1spm/use_${dprefix}/${dname}.src
fi

echo "*** Paths to training data and dictionary ***"
echo "| train_json_dir: ${train_json_dir}"
echo "| val_json_dir: ${val_json_dir}"
echo "| source dictionary: ${dict_src}"
echo "| target dictionary: ${dict_tgt}"

# Find the latest snapshot (if it exists)
resume_dir=$expdir/results
exist_snaphots=false
for i in $resume_dir/snapshot.iter.*; do test -f "$i" && exist_snaphots=true && break; done
if [ "${exist_snaphots}" = true ]; then
    ckpt_nums=$(ls $resume_dir | grep snapshot | sed 's/[^0-9]*//g' | sed 's/\n/" "/g')
    last_ep=$(echo "${ckpt_nums[*]}" | sort -nr | head -n1)
    resume=${resume_dir}/snapshot.iter.${last_ep}
    echo "Last snapshot: snapshot.iter.${last_ep}"
fi

if [[ ${stage} -le 4 ]] && [[ ${stop_stage} -ge 4 ]]; then
    echo "***** stage 4: Network Training *****"

    # Resume training
    if [[ -z ${resume} ]]; then
        echo "Training from scratch!"
    else
        echo "Resume training from ${resume}"
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
        --dict-src ${dict_src} \
        --dict-tgt ${dict_tgt} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume $resume \
        --train-json ${train_json_dir} \
        --valid-json ${val_json_dir} \
        --early-stop-criterion ${early_stop_criterion} \
        --use-lid ${use_lid} \
        --do-st ${do_st} \
        --report-bleu \
        --init-from-decoder-asr ${init_from_decoder_asr} \
        --init-from-decoder-mt ${init_from_decoder_mt} \
        --use-adapters ${use_adapters} \
        --train-adapters ${train_adapters} \
        --use-multi-dict ${use_multi_dict} \
        --use-adapters-for-asr ${use_adapters_for_asr} \
        --use-adapters-in-enc ${use_adapters_in_enc}
        # --enc-init ${asr_model} \
        # --dec-init ${st_model}
    
    echo "Log output is saved in ${expdir}/train.log"
fi


if [[ ${stage} -le 5 ]] && [[ ${stop_stage} -ge 5 ]]; then
    echo "***** stage 5: Decoding *****"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        if [[ -z ${trans_model} ]]; then
            if (( ${max_iter_eval} >= ${last_ep} )); then
                max_iter_eval=${last_ep}
            fi

            # model used to translate
            if (( ${n_average} == 1 )); then
                trans_model=model.acc.best.upto${max_iter_eval}
                opt="--log ${expdir}/results/log"
            else
                if ${use_valbest_average}; then
                    trans_model=model.val${n_average}.avg.best.upto${max_iter_eval}
                    opt="--log ${expdir}/results/log"
                else
                    trans_model=model.last${n_average}.avg.best.upto${max_iter_eval}
                    opt="--log"
                fi
            fi
            echo "| trans_model: ${trans_model}"

            if [[ ! -f ${expdir}/results/${trans_model} ]]; then
                echo "*** Get trans_model ***"
                local/average_checkpoints_st.py \
                    ${opt} \
                    --max-iter-eval ${max_iter_eval} \
                    --min-iter-eval ${min_iter_eval} \
                    --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.iter.* \
                    --out ${expdir}/results/${trans_model} \
                    --num ${n_average}        
            else
                echo "| trans_model ${expdir}/results/${trans_model} existed."
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
        decode_config_lg_pair=${decode_config_dir}/${decode_config}.${lg_pair}.yaml
        decode_dir=decode_$(basename ${train_config%.*})_$(basename ${decode_config})_${split}_${lg_pair}_${trans_model}
        feat_trans_dir=${datadir}/${split}
        echo "| decode_dir: ${decode_dir}"
        echo "| feat_trans_dir: ${feat_trans_dir}"

        # split data
        if [ ! -f "${feat_trans_dir}/split${nj}utt_${tgt_langs}/${lg_pair}.${nj}.json" ]; then
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
        if [[ $tag != *"asr_model"* ]]; then
            echo "Compute BLEU..."
            chmod +x local/score_bleu_st.sh
            echo "BPE model: ${bpemodel_tgt}"
            local/score_bleu_st.sh --case ${tgt_case} \
                                   --bpe ${nbpe} --bpemodel ${bpemodel_tgt}.model \
                                   ${expdir}/${decode_dir} ${lg_tgt} ${dict_tgt} ${dict_src} \
                                   ${remove_non_verbal_eval}
            cat ${expdir}/${decode_dir}/result.tc.txt
        fi

        # Compute WER
        if [[ $tag != *"mt_model"* ]] && [[ $tag != *"no_asr"* ]]; then
            echo "Compute WER score..."
            idx=1
            if [[ $tag == *"asr_model"* ]]; then
                idx=0
            fi
            local/score_sclite_st.sh --case ${src_case} --wer true \
                                     --bpe ${nbpe_src} --bpemodel ${bpemodel_src}.model \
                                     ${expdir}/${decode_dir} ${dict_src} ${idx}
            cat ${expdir}/${decode_dir}/result.wrd.wer.txt
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished decoding."
fi