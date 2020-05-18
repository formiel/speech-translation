#!/bin/bash
# Copyright 2019 Hang Le (hangtp.le@gmail.com)

#SBATCH --job-name=mc-pt    # nom du job
#SBATCH --account=dbn@gpu
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p2                    # demande d'allocation sur la partition GPU
#SBATCH --exclusive
#SBATCH --ntasks=8                 # nombre de tâche MPI
#SBATCH --ntasks-per-node=8         # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:8                # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=3            # nombre de coeurs à réserver par tâche
#SBATCH --hint=nomultithread              
#SBATCH --time=35:00:00              # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=./slurm_logs/train_de_nl_%j.out # nom du fichier de sortie

# Load necessary modules
module load python/3.7.5 cuda/10.1.1 cudnn/7.6.5.32-cuda-10.1 nccl/2.6.4-1-cuda gcc/7.3.0 openmpi/4.0.2-cuda

# echo des commandes lancées
set -x

cd $WORK/espnet/egs/must_c/st_multilingual

ngpu=8
stage=4
stop_stage=4

tag=$1

# Check number of arguments
if [ $# -eq 1 ]; then
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

./run-multi.sh  --ngpu ${ngpu} --stage $stage --stop-stage $stop_stage \
        --tag $tag \
        --tgt-lang $tgt_lang \
        --train-config ${train_config} \
        --preprocess-config ${preprocess_config}