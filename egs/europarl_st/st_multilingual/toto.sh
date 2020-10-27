#!/bin/bash

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode de dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode es dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode fr dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode it dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode nl dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode pt dev 1 2500000000

sbatch submit-p1-decode.slurm dict1_bpe8k_1decoder_pre_use_adapters_for_asr exp separate_decode ro dev 1 2500000000