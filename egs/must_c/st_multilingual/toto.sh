#!/bin/bash

# dir=$WORK/Pretrained_models/MUSTC
# for folder in $dir/*; do
#     if [[ ${folder} == *".gz" ]]; then
#         basename=$(basename ${folder})
#         if [[ ${basename} == "train_sp."* ]]; then
#             extracted="asr_${basename::-7}"
#         fi
#         if [[ ${basename} == "transformer."* ]]; then
#             extracted="mt_${basename::-7}"
#         fi
#         echo "Extracting ${folder} to ${dir}/${extracted}..."
#         mkdir -p ${dir}/${extracted}
#         tar -xvf ${folder} -C ${dir}/${extracted}
#     fi
# done

sbatch submit.slurm decoder_pre_1decoder separate_decode 150 tst-COMMON.en-it.it

sbatch submit.slurm decoder_pre_1decoder separate_decode 150 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_1decoder separate_decode 150 tst-COMMON.en-pt.pt

sbatch submit.slurm decoder_pre_1decoder separate_decode 150 tst-COMMON.en-ro.ro

sbatch submit.slurm decoder_pre_1decoder separate_decode 150 tst-COMMON.en-ru.ru

sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode_separate 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode_separate 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode_separate 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode_separate 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode_separate 42 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode_separate 42 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 33 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 33 tst-COMMON.en-nl.nl


sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode 42 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode 42 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode 33 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode 33 tst-COMMON.en-nl.nl


sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode_weighting 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xself_xsum_waitk0_lang_de_nl common_decode_weighting 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode_weighting 41 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xsum_waitk0_lang_de_nl common_decode_weighting 41 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode_weighting 42 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl common_decode_weighting 42 tst-COMMON.en-nl.nl

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_weighting 33 tst-COMMON.en-de.de

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_weighting 33 tst-COMMON.en-nl.nl

