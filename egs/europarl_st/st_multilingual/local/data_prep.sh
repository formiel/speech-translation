#!/bin/bash

# Modified by Hang Le
# The original copyright is appended below
# --
# Copyright 2019 Kyoto University (Hirofumi Inaguma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <src-dir> <src-lang> <tgt-lang>"
    echo "e.g.: $0 Data/europarl_st src_lang tgt_lang"
    exit 1;
fi

src_lang=$2
tgt_lang=$3

for set in train dev test; do
    src=$1/${src_lang}/${tgt_lang}/${set}
    dst=data/local/${src_lang}-${tgt_lang}/${set}

    [ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

    wav_dir=$1/${src_lang}/audios
    lst=${src}/segments.lst
    recog=${src}/segments.${src_lang}
    tgt=${src}/segments.${tgt_lang}

    mkdir -p ${dst} || exit 1;

    [ ! -d ${wav_dir} ] && echo "$0: no such directory ${wav_dir}" && exit 1;
    [ ! -f ${lst} ] && echo "$0: expected file ${lst} to exist" && exit 1;
    [ ! -f ${recog} ] && echo "$0: expected file ${recog} to exist" && exit 1;
    [ ! -f ${tgt} ] && echo "$0: expected file ${tgt} to exist" && exit 1;

    wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
    trans_recog=${dst}/text.${src_lang}; [[ -f "${trans_recog}" ]] && rm ${trans_recog}
    trans_tgt=${dst}/text.${tgt_lang}; [[ -f "${trans_tgt}" ]] && rm ${trans_tgt}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

    # sanity check
    n=$(cat ${lst} | wc -l)
    n_recog=$(cat ${recog} | wc -l)
    n_tgt=$(cat ${tgt} | wc -l)
    [ ${n} -ne ${n_recog} ] && echo "Warning: expected ${n} data data files, found ${n_recog}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data data files, found ${n_tgt}" && exit 1;

    # (1a) Transcriptions and translations preparation
    # make basic transcription file (add segments info)
    cp ${lst} ${dst}/.lst0
    awk '{
        spkid=$1; offset=$2; end=$3;
        duration=sprintf("%.7f", end-offset);
        if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
        else extendt=0;
        offset=sprintf("%.7f", offset);
        if ( offset > extendt ) startt=offset-extendt;
        else startt=offset;
        endt=offset+duration+extendt;
        printf("%s_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
    }' ${dst}/.lst0 > ${dst}/.lst1
    # NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them

    cp ${recog} ${dst}/${src_lang}.org
    cp ${tgt} ${dst}/${tgt_lang}.org

    for lang in ${src_lang} ${tgt_lang}; do
        # normalize punctuation
        normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        local/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        # tokenization
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

        paste -d " " ${dst}/.lst1 ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/.lst1 ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/.lst1 ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done


    # sanity check
    n=$(cat ${dst}/.lst1 | wc -l)
    n_recog=$(cat ${dst}/${src_lang}.norm.tc.tok | wc -l)
    n_tgt=$(cat ${dst}/${tgt_lang}.norm.tc.tok | wc -l)
    [ ${n} -ne ${n_recog} ] && echo "Warning: expected ${n} data data files, found ${n_recog}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data data files, found ${n_tgt}" && exit 1;


    # (1c) Make segments files from transcript
    #segments file format is: utt-id start-time end-time, e.g.:
    #ted_00001_0003501_0003684 ted_0001 003.501 0003.684
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1]; startf=S[2]; endf=S[3];
        printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/segments

    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1];
        printf("%s cat '${wav_dir}'/%s.wav |\n", spkid, S[1]);
    }' < ${dst}/text.tc.${tgt_lang} | uniq | sort > ${dst}/wav.scp

    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1]; print $1 " " spkid
    }' ${dst}/segments | uniq | sort > ${dst}/utt2spk

    cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt

    # sanity check
    n_recog=$(cat ${dst}/text.tc.${src_lang} | wc -l)
    n_tgt=$(cat ${dst}/text.tc.${tgt_lang} | wc -l)
    [ ${n_recog} -ne ${n_tgt} ] && echo "Warning: expected ${n_recog} data data files, found ${n_tgt}" && exit 1;

    # Copy stuff into its final locations [this has been moved from the format_data script]
    mkdir -p data/${set}.${src_lang}-${tgt_lang}

    # remove duplicated utterances (the same offset)
    echo "remove duplicate lines..."
    cut -d ' ' -f 1 ${dst}/text.tc.${src_lang} | sort | uniq -c | sort -n -k1 -r \
        | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
    cut -d ' ' -f 1 ${dst}/text.tc.${src_lang} | sort | uniq -c | sort -n -k1 -r \
        | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
    reduce_data_dir.sh ${dst} ${dst}/reclist data/${set}.${src_lang}-${tgt_lang}
    for l in ${src_lang} ${tgt_lang}; do
        for case in tc lc lc.rm; do
            cp ${dst}/text.${case}.${l} data/${set}.${src_lang}-${tgt_lang}/text.${case}.${l}
        done
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.${src_lang} text.lc.${src_lang} text.lc.rm.${src_lang} text.tc.${tgt_lang} text.lc.${tgt_lang} text.lc.rm.${tgt_lang}" \
        data/${set}.${src_lang}-${tgt_lang}

    # sanity check
    n_seg=$(cat data/${set}.${src_lang}-${tgt_lang}/segments | wc -l)
    n_text=$(cat data/${set}.${src_lang}-${tgt_lang}/text.tc.${tgt_lang} | wc -l)
    [ ${n_seg} -ne ${n_text} ] && echo "Warning: expected ${n_seg} data data files, found ${n_text}" && exit 1;

    echo "$0: successfully prepared data in ${dst}"
done
