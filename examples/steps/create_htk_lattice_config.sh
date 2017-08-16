#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo usage: $0 "<n_units> <n_states> <outdir>"
    exit 1
fi

n_units=$1
n_states=$2
outdir=$3

if [ ! -e ${outdir}/.done ] ; then

    mkdir -p ${outdir}

    echo "TARGETKIND     = USER" > ${outdir}/HVite.cfg

    # Create a pseudo phonemes list. The number of phonemes is defined by the
    # n. of units in the phone loop model.
    for p in `seq 0 $((n_units - 1))`; do
    	echo 'a'${p} >> ${outdir}/phonemes
	    for s in `seq $n_states`; do
	        echo 'a'${p}__${s} >>  ${outdir}/states
	    done
    done
    cp ${outdir}/phonemes ${outdir}/hmmlist
    cat ${outdir}/hmmlist | awk '{print $1,$1}' > ${outdir}/dict

    # Create recognition net.
    HBuild ${outdir}/hmmlist ${outdir}/monophones_lnet.hvite

    # Create the HTK HMM definitions file.
    steps/create_HMM_def.sh ${outdir}/states ${outdir}/hmmdefs.hvite || exit 1

    date > ${outdir}/.done
fi
