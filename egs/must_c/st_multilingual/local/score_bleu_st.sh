#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
case=lc
set=""

. utils/parse_options.sh

if [ $# -lt 3 ]; then
    echo "Usage: $0 <decode-dir> <tgt_lang> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
tgt_lang=$2
dic_tgt=$3
dic_src=$4

if [[ ! -f ${dir}/data.json ]]; then
    concatjson.py ${dir}/data.*.json > ${dir}/data.json
    echo "Finished concatenating json files."
else
    echo "json files have been concatenated."
fi

json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org.${tgt_lang} \
    --hyps ${dir}/hyp.trn.org.${tgt_lang} --srcs ${dir}/src.trn.org.${tgt_lang} --dict-src ${dic_src}

# remove uttterance id
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/ref.trn.org.${tgt_lang} > ${dir}/ref.trn.${tgt_lang}
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/hyp.trn.org.${tgt_lang} > ${dir}/hyp.trn.${tgt_lang}
perl -pe 's/\([^\)]+\)\n/\n/g;' ${dir}/src.trn.org.${tgt_lang} > ${dir}/src.trn.${tgt_lang}

if [ -n "$bpe" ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn.${tgt_lang} | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn.${tgt_lang}
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn.${tgt_lang} | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn.${tgt_lang}
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn.${tgt_lang} | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn.${tgt_lang}
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref.trn.${tgt_lang} > ${dir}/ref.wrd.trn.${tgt_lang}
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.trn.${tgt_lang} > ${dir}/hyp.wrd.trn.${tgt_lang}
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/src.trn.${tgt_lang} > ${dir}/src.wrd.trn.${tgt_lang}
fi

# detokenize
detokenizer.perl -l ${tgt_lang} -q < ${dir}/ref.wrd.trn.${tgt_lang} > ${dir}/ref.wrd.trn.detok.${tgt_lang}
detokenizer.perl -l ${tgt_lang} -q < ${dir}/hyp.wrd.trn.${tgt_lang} > ${dir}/hyp.wrd.trn.detok.${tgt_lang}
detokenizer.perl -l ${tgt_lang} -q < ${dir}/src.wrd.trn.${tgt_lang} > ${dir}/src.wrd.trn.detok.${tgt_lang}

# remove language IDs
if [ -n "${nlsyms}" ]; then
    cp ${dir}/ref.wrd.trn.detok.${tgt_lang} ${dir}/ref.wrd.trn.detok.tmp.${tgt_lang}
    cp ${dir}/hyp.wrd.trn.detok.${tgt_lang} ${dir}/hyp.wrd.trn.detok.tmp.${tgt_lang}
    cp ${dir}/src.wrd.trn.detok.${tgt_lang} ${dir}/src.wrd.trn.detok.tmp.${tgt_lang}
    filt.py -v $nlsyms ${dir}/ref.wrd.trn.detok.tmp.${tgt_lang} > ${dir}/ref.wrd.trn.detok.${tgt_lang}
    filt.py -v $nlsyms ${dir}/hyp.wrd.trn.detok.tmp.${tgt_lang} > ${dir}/hyp.wrd.trn.detok.${tgt_lang}
    filt.py -v $nlsyms ${dir}/src.wrd.trn.detok.tmp.${tgt_lang} > ${dir}/src.wrd.trn.detok.${tgt_lang}
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.wrd.trn.detok.${tgt_lang}
    sed -i.bak3 -f ${filter} ${dir}/ref.wrd.trn.detok.${tgt_lang}
    sed -i.bak3 -f ${filter} ${dir}/src.wrd.trn.detok.${tgt_lang}
fi
# NOTE: this must be performed after detokenization so that punctuation marks are not removed

if [ ${case} = tc ]; then
    echo ${set} > ${dir}/result.tc.txt
    multi-bleu-detok.perl ${dir}/ref.wrd.trn.detok.${tgt_lang} < ${dir}/hyp.wrd.trn.detok.${tgt_lang} >> ${dir}/result.tc.txt
    echo "write a case-sensitive BLEU result in ${dir}/result.tc.txt"
    cat ${dir}/result.tc.txt
else
    echo ${set} > ${dir}/result.lc.txt
    multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn.detok.${tgt_lang} < ${dir}/hyp.wrd.trn.detok.${tgt_lang} > ${dir}/result.lc.txt
    echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
    cat ${dir}/result.lc.txt
fi