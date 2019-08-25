#!/bin/bash

SIZE_LR=1
HS1=150 #
HL1=3 #
WS=9
SE=2 
BIDIR1=1 #
NORM=ln #

BATCH=64 
FEAT1=SN_SF1_SIN_SF1_ZN_F9
FEAT_NUM1=9

START_EPOCH=0 
END_EPOCH=20 

PRETRAIN=False 

LOG_FILE="log/sdt6_resnet_semi_${WS}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample.log"

mkdir -p log

echo -e "Saving Record to ${LOG_FILE}"

bash script/train_TONAS_5class_resnet_semi.sh ${SIZE_LR} ${WS} ${SE} ${BATCH} ${FEAT1} ${FEAT_NUM1} ${START_EPOCH} ${END_EPOCH} ${PRETRAIN}| tee ${LOG_FILE}

echo -e "Saving Record to ${LOG_FILE}"