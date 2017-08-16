#!/bin/bash
#$ -N TRAIN_PHONE_LOOP
#$ -j y
#$ -cwd
#$ -o TRAIN_PHONE_LOOP.log
#$ -pe smp 10
#$ -l mem_free=1G,ram_free=1G

source /homes/kazi/iondel/.bashrc
cd /mnt/matylda5/iondel/workspace/2017/JSALT/TIMIT

#########################
# Global configuration. #
#########################
log_level=debug
profile=default
njobs=4
delay=30

#####################
# Model definition. #
#####################
n_units=48
n_states=3
n_comp=4
features=MBN
sample_var=0.25

###########################
# Training configuration. #
###########################
epochs=20
lrate_hmm=1e-1
batch_size=400
transcription=data/train/transcriptions_idxs.txt

######################
# Output directories #
######################
out_dir=$PWD/ploop_u${n_units}_s${n_states}_c${n_comp}_${features}
tmp_dir=${out_dir}/tmp
mkdir -p ${out_dir}
mkdir -p ${tmp_dir}

python utils/train_phone_loop.py \
    --log_level ${log_level} \
    --profile ${profile} \
    --njobs ${njobs} \
    --delay ${delay} \
    --n_units ${n_units} \
    --n_states ${n_states} \
    --n_comp ${n_comp} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --lrate_hmm ${lrate_hmm} \
    --transcription ${transcription} \
    --sample_var ${sample_var} \
    data/train/${features}/fea_list.txt "$tmp_dir" "${out_dir}/model.bin" \
    "${out_dir}/stats.bin"

