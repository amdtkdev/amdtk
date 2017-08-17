#!/bin/env bash

##############################################################################
## SETTINGS

#### LOGGING LEVEL ####
log_level=debug                 # logging level (debug, info, warning, error)

#### PARALLEL ENVIRONMENT ####
profile=sge                      # ipyparallel environment name
njobs=10                         # number of jobs to use

#### FEATURES ####
fea_type=mbn                    # type of the features
fea_format=htk                  # features file format
n_stacked_frames=0              # number of stacked frames
mv_norm=true                    # mean/variance normalization
fea_dname=${fea_type}_sf${n_stacked_frames}
if [ $mv_norm = true ]; then
    fea_dname=${fea_dname}_mvnorm
fi                              # features level directory name

#### PHONE-LOOP MODEL ####
n_units=$(cat data/idx_phone_map.txt | wc -l) || exit 1
                                # number of units (i.e. phone/pseudo-phone)
n_states=3                      # number of HMM states per unit
n_comp=4                        # number of Gaussian components per HMM state
model_dname=ploop_u${n_units}_s${n_states}_c${n_comp}
                                # model level directory name

#### TRAINING ####
train_set=train                 # training subset
epochs=50                       # number of epochs
batch_size=400                  # size of the mini-batches (in utterance)
hmm_lrate=1e-1                  # learning rate for the HMM parameters
ae_lrate=1e-3                   # learning rate for the auto-encoder parameters
transcription=data/${train_set}/transcription_idx.txt
                                # transcription (leave empty for unit discovery)
train_dname=${train_set}_e${epochs}_bs${batch_size}_hlr${hmm_lrate}_aelr${autoencoder_lrate}
                                # training level directory name

#### DECODING ####
decode_sets="train test"        # list of subsets to decode

#### LATTICE GENERATION ####
lattice_sets="train test"       # list of subsets to generate lattices

#### OUTPUT ####
out_dir=${PWD}/${model_dname}/${fea_dname}
                                # main output directory

##############################################################################


# Step 1:
# Prepare the features loader and compute the statistics of the training data.
#

if [ ! -f ${out_dir}/fea_loader/.done ]; then

    echo "==================================================================="
    echo "                       Prepare training                            "
    echo "==================================================================="

    # Create output directory.
    mkdir -p ${out_dir}/fea_loader

    # Transforming boolean variable into command option.
    if [ ${mv_norm} = true ]; then
        apply_mv_norm="--mv_norm"
    else
        apply_mv_norm=""
    fi

    python utils/create_features_loader.py \
        --log_level ${log_level} \
        --profile ${profile} \
        --njobs ${njobs} \
        --file_format ${fea_format} \
        ${apply_mv_norm} \
        --context ${n_stacked_frames} \
        data/${train_set}/${fea_type}/fea_list.txt \
        ${out_dir}/fea_loader/data_stats.bin \
        ${out_dir}/fea_loader/fea_loader.bin || exit 1

    date > ${out_dir}/fea_loader/.done
fi


# Step 2:
# Create the initial model.
#

if [ ! -f ${out_dir}/init_model/.done ]; then

    echo "==================================================================="
    echo "                         Create initial Model                      "
    echo "==================================================================="

    # Create the output directory.
    mkdir -p ${out_dir}/init_model

    python utils/create_phone_loop.py \
        --log_level ${log_level} \
        --n_units ${n_units} \
        --n_states ${n_states} \
        --n_comp ${n_comp} \
        ${out_dir}/fea_loader/data_stats.bin \
        ${out_dir}/init_model/model.bin || exit 1

    date > ${out_dir}/init_model/.done
fi


# Step 3:
# Train the model.
#

if [ ! -f ${out_dir}/training/.done ]; then

    echo "==================================================================="
    echo "                         Training                                  "
    echo "==================================================================="

    # Create the output directory.
    mkdir -p ${out_dir}/training

    if [ ! -z ${transcription} ]; then
        use_transcription="--transcription ${transcription}"
    else
        use_transcription=""
    fi

    python utils/train_model.py \
        --log_level ${log_level} \
        --profile ${profile} \
        --njobs ${njobs} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --lrate_hmm ${hmm_lrate} \
        --lrate_autoencoder ${ae_lrate} \
        ${use_transcription} \
        ${out_dir}/training \
        data/${train_set}/${fea_type}/fea_list.txt \
        ${out_dir}/fea_loader/fea_loader.bin \
        ${out_dir}/fea_loader/data_stats.bin \
        ${out_dir}/init_model/model.bin \
        ${out_dir}/training/model.bin || exit 1

    date > ${out_dir}/training/.done
fi


# Step 4:
# Decode the data.
#

for subset in ${decode_sets}; do

    if [ ! -f ${out_dir}/decode/${subset}/labels/.done ]; then

        echo "==================================================================="
        echo "                         Decoding $subset set                      "
        echo "==================================================================="

        # Create the output directory.
        mkdir -p ${out_dir}/decode/${subset}/labels

        python utils/decode.py \
            --log_level ${log_level} \
            --profile ${profile} \
            --njobs ${njobs} \
            data/idx_phone_map.txt \
            ${out_dir}/fea_loader/fea_loader.bin \
            ${out_dir}/fea_loader/data_stats.bin \
            ${out_dir}/training/model_45.bin \
            data/${subset}/${fea_type}/fea_list.txt \
            ${out_dir}/decode/${subset}/labels || exit 1

        # Gather all the transcription into a single file.
        find ${out_dir}/decode/${subset}/labels -name '*lab' \
            -exec cat {} \; \
            > ${out_dir}/decode/${subset}/labels/transcription.txt || exit 1

        date > ${out_dir}/decode/${subset}/labels/.done
    fi
done


# Step 5:
# Compute the phone error rate.
#

for subset in ${decode_sets}; do

    if [ ! -f ${out_dir}/decode/${subset}/per/.done ]; then

        echo "==================================================================="
        echo "          Computing phone error rate for $subset set               "
        echo "==================================================================="

        # Create the output directory.
        mkdir -p ${out_dir}/decode/${subset}/per

        python utils/per.py \
            --log_level ${log_level} \
            --profile ${profile} \
            --njobs ${njobs} \
            data/ref_phone_map.txt \
            data/hyp_phone_map.txt \
            data/${subset}/transcription.txt \
            ${out_dir}/decode/${subset}/labels/transcription.txt || exit 1

        date > ${out_dir}/decode/${subset}/per/.done
    fi
done

