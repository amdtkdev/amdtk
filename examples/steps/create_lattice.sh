#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo 'usage:' $0 '<conf_latt_dir> <posteriors_htk> <out_dir>'
    exit
fi

conf_latt_dir=$1
pos_file=$2
out_dir=$3
penalty=-1
gscale=1
beam_thresh=0.0

base1=`basename ${pos_file}`
base_name=${base1%.*}

mkdir -p ${out_dir}

HVite -T 1 -y 'lab' -z 'lat' -l ${out_dir} \
    -C ${conf_latt_dir}/HVite.cfg   \
    -w ${conf_latt_dir}/monophones_lnet.hvite \
    -n 2 1 \
    -p ${penalty} \
    -q 'Atval' \
    -s ${gscale} \
    -t ${beam_thresh} \
    -H ${conf_latt_dir}/hmmdefs.hvite \
    ${conf_latt_dir}/dict \
    ${conf_latt_dir}/phonemes \
    $pos_file || exit 1

