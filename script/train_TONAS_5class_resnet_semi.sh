#!/bin/bash

DHEAD="data/TONAS_note/${9}/"
AHEAD1="ans/TONAS_SDT6/sdt6_"
UHEAD="data/Feat/" #"data/TONAS_note/${9}/"
BASE_LR=0.001
SIZE_LR=$1
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
WS=$2
SE=$3

BATCH=$4
FEAT1=$5
FEAT_NUM1=$6
TRAINCOUNT=71

START_EPOCH=$7
END_EPOCH=$8

PRETRAIN=$9

MDIR="model/5class_resnet_${WS}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample"
DMFILE1="${MDIR}/onoffset_ondec_k${WS}e${END_EPOCH}b${BATCH}_${FEAT1}"
TRDMFILE1="${MDIR}/onoffset_ondec_k${WS}e${END_EPOCH}b${BATCH}_${FEAT1}_train"
LFILE="loss/sdt6_resnet_onoffset_${WS}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample.csv"

PRETRAINFILE="baseline_models/PyramidNet_FreqMask_PitchShift_Baseline"

echo -e "Training OnOffset Model Exp2 Info:"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${END_EPOCH} BATCH=${BATCH} FEAT1=${FEAT1}"
echo -e "Onset Decoder Model: ${DMFILE1}"
echo -e "Loss: ${LFILE}"

mkdir -p loss
mkdir -p model
mkdir -p ${MDIR}

for e in $(seq ${START_EPOCH} $((${END_EPOCH}-1)))
do
    for num in $(seq 1 82) 
    do
        python3 src/onoffset_5class_resnet_semi.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} \
        -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
        --window-size ${WS} --single-epoch ${SE} \
        --loss-record ${LFILE} --batch-size ${BATCH} --feat1 ${FEAT_NUM1} \
        -dmt1 ${TRDMFILE1} -u1 ${UHEAD} -pretrain_model ${PRETRAIN} -pretrain_dest ${PRETRAINFILE}
    done
    
done