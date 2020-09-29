#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
case=lc.rm

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <data-dir> <dict> <output-idx>";
    exit 1;
fi

dir=$1
dic=$2
idx=$3

if [[ ! -f ${dir}/data.json ]]; then
    concatjson.py ${dir}/data.*.json > ${dir}/data.json
    echo "Finished concatenating json files."
else
    echo "json files have already been concatenated."
fi
json2trn.py ${dir}/data.json ${dic} --refs ${dir}/ref.trn.en --hyps ${dir}/hyp.trn.en \
                                    --output-idx ${idx}

if ${remove_blank}; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn.en
fi
if [ -n "${nlsyms}" ]; then
    cp ${dir}/ref.trn.en ${dir}/ref.trn.org.en
    cp ${dir}/hyp.trn.en ${dir}/hyp.trn.org.en
    filt.py -v ${nlsyms} ${dir}/ref.trn.org.en > ${dir}/ref.trn.en
    filt.py -v ${nlsyms} ${dir}/hyp.trn.org.en > ${dir}/hyp.trn.en
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn.en
    sed -i.bak3 -f ${filter} ${dir}/ref.trn.en
fi

# lowercasing
lowercase.perl < ${dir}/hyp.trn.en > ${dir}/hyp.trn.lc.en
lowercase.perl < ${dir}/ref.trn.en > ${dir}/ref.trn.lc.en

# remove punctuation
paste -d "(" <(cut -d '(' -f 1 ${dir}/hyp.trn.lc.en | local/remove_punctuation.pl | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/hyp.trn.lc.en) > ${dir}/hyp.trn.lc.rm.en
paste -d "(" <(cut -d '(' -f 1 ${dir}/ref.trn.lc.en | local/remove_punctuation.pl | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/ref.trn.lc.en) > ${dir}/ref.trn.lc.rm.en

# detokenize
detokenizer.perl -l en -q < ${dir}/ref.trn.lc.rm.en > ${dir}/ref.trn.lc.rm.detok.en
detokenizer.perl -l en -q < ${dir}/hyp.trn.lc.rm.en > ${dir}/hyp.trn.lc.rm.detok.en

sclite -r ${dir}/ref.trn.lc.rm.detok.en trn -h ${dir}/hyp.trn.lc.rm.detok.en trn -i rm -o all stdout > ${dir}/result.cer.txt

echo "write a CER (or TER) result in ${dir}/result.cer.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.cer.txt

if ${wer}; then
    if [ -n "$bpe" ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn.lc.rm.en | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn.lc.rm.en
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn.lc.rm.en | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn.lc.rm.en
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn.lc.rm.en > ${dir}/ref.wrd.trn.lc.rm.en
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn.lc.rm.en > ${dir}/hyp.wrd.trn.lc.rm.en
    fi

    # detokenize
    detokenizer.perl -l en -q < ${dir}/ref.wrd.trn.lc.rm.en > ${dir}/ref.wrd.trn.lc.rm.detok.en
    detokenizer.perl -l en -q < ${dir}/hyp.wrd.trn.lc.rm.en > ${dir}/hyp.wrd.trn.lc.rm.detok.en

    sclite -r ${dir}/ref.wrd.trn.lc.rm.detok.en trn -h ${dir}/hyp.wrd.trn.lc.rm.detok.en trn -i rm -o all stdout > ${dir}/result.wrd.wer.txt

    echo "write a WER result in ${dir}/result.wrd.wer.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.wer.txt
fi