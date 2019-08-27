#!/bin/bash

DHEAD="data/ISMIR2014_note/"
AHEAD="ans/ISMIR2014_ans/"
PHEAD="pitch/ISMIR2014/"
TDHEAD="data/TONAS_note/"
TAHEAD="ans/TONAS_ans/"
TPHEAD="pitch/TONAS/"
SIZE_LR=1
BASE_LR=0.001
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
WS=9
SE=1 #

END_EPOCH=20 
BATCH=64 
FEAT=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM=9

THRESHOLD=0.5

MDIR="model/5class_resnet_${WS}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample"
DMFILE1="${MDIR}/onoffset_ondec_k${WS}e${END_EPOCH}b${BATCH}_${FEAT1}"
EFILE="output/single/ISMIR2014_5class_resnet_k${WS}_e${END_EPOCH}b${BATCH}_${FEAT}.csv"
VFILE="output/total/ISMIR2014_5class_resnet_k${WS}_e${END_EPOCH}b${BATCH}_${FEAT}.csv"
TEFILE="output/train_single/ISMIR2014_5class_resnet_k${WS}_e${END_EPOCH}b${BATCH}_${FEAT}_sample.csv"
TVFILE="output/train_total/ISMIR2014_5class_resnet_k${WS}_e${END_EPOCH}b${BATCH}_${FEAT}_sample.csv"
TROUTDIR="output/5class_resnet_est"

echo -e "Evaluating OnOffset Model Info:"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${END_EPOCH} BATCH=${BATCH} FEAT=${FEAT}"
echo -e "Onset Decoder Model: ${DMFILE1}"

mkdir -p ${TROUTDIR}
mkdir -p output/single
mkdir -p output/total
mkdir -p output/train_single
mkdir -p output/train_total

echo -e "Start Evaluation on ISMIR2014 Validation Set"
for num in $(seq 1 38)
do
    python3 src/eval_5class_resnet_fmeasure.py \
    -d ${DHEAD}/${FEAT}/${num}_${FEAT} -a ${AHEAD}${num}.GroundTruth -pf ${PHEAD}${num}_P \
    -dm1 ${DMFILE1} -p ${num} -ef ${EFILE} -tf ${VFILE} -l ${LR} \
    --ws ${WS} --single-epoch ${SE} --feat ${FEAT_NUM} --threshold ${THRESHOLD} \
    -of ${TROUTDIR}/${num}_test -sf ${TROUTDIR}/${num}_sdt_test -sm ${TROUTDIR}/${num}_sm_test
done