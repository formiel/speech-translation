#!/bin/bash
# Copyright 2019 Hang Le (hangtp.le@gmail.com)

#SBATCH --job-name=shared    # nom du job
#SBATCH --account=dbn@gpu
#SBATCH --partition=prepost                    # demande d'allocation sur la partition GPU
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96            # nombre de coeurs à réserver par tâche   
#SBATCH --time=5:00:00              # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=./slurm_logs/decode-%j.txt # nom du fichier de sortie

# Load necessary modules
# module load pytorch-gpu/py3/1.4.0
module load python/3.7.5 cuda/10.1.1 cudnn/7.6.5.32-cuda-10.1 nccl/2.6.4-1-cuda gcc/7.3.0 openmpi/4.0.2-cuda

# echo des commandes lancées
set -x

cd $WORK/espnet/egs/must_c/st_multilingual

stage=5
ngpu=0

tag=$1
decode=$2
decode_config=./conf/tuning/${decode}

iter=$3
if [[ $iter -gt 0 ]]; then
    trans_model=model.${iter}k.acc.best
elif [[ $iter -eq 0 ]]; then
    trans_model=model.acc.best
else
    trans_model=""
fi
echo "*** trans_model: ${trans_model} ***"

trans_set=$4

bash run-cmd.sh $tag $stage $ngpu $decode_config $trans_model $trans_set