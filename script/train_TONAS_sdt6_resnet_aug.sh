#!/bin/bash

DHEAD="data/TONAS_note/${9}/"
AHEAD1="ans/TONAS_SDT6/sdt6_"
BASE_LR=0.001
SIZE_LR=$1
LR=$(echo "${BASE_LR} * ${SIZE_LR}" | bc)
HS1=$2
HL1=$3
WS=$4
SE=$5
BIDIR1=$6
NORM=$7

BATCH=$8
FEAT1=$9
FEAT_NUM1=${10}
TRAINCOUNT=71

START_EPOCH=${11}
END_EPOCH=${12}

MDIR="model/sdt6_resnet_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample"
EMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT1}"
DMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT1}"
TREMFILE1="${MDIR}/onoffset_attn_${NORM}_onenc_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT1}_train"
TRDMFILE1="${MDIR}/onoffset_attn_${NORM}_ondec_k${WS}l${HL1}h${HS1}b${BIDIR1}e${END_EPOCH}b${BATCH}_${FEAT1}_train"
LFILE="loss/sdt6_resnet50_onoffset_attn_${NORM}${WS}_l${HL1}h${HS1}b${BIDIR1}_e${END_EPOCH}b${BATCH}_${FEAT1}_sample.csv"

BUCKETLIST=('' '5' '8' '29' '6' '14' '31' '18' '1' '7' '16' '3' '4' '11' '67' '9' '41' '12' '28' '2' '72' '21' '79' '36' '81' '45' '56' '76' '42' '74' '27' '63' '33' '44' '40' '38' '25' '59' '46' '52' '32' '30' '53' '71' '80' '50' '58' '57' '37' '61' '65' '73' '39' '47' '24' '82' '51' '34' '66' '43' '78' '49' '69' '75' '35' '70' '54' '48' '68' '26' '77' '64' '62' '13' '10' '60' '55' '15' '17' '19' '20' '22' '23' '0')

echo -e "Training OnOffset Model Exp2 Info:"
echo -e "HS1=${HS1} HL1=${HL1} BIDIR1=${BIDIR1} NORM=${NORM}"
echo -e "WS=${WS} SE=${SE}"
echo -e "EPOCHS=${END_EPOCH} BATCH=${BATCH} FEAT1=${FEAT1}"
echo -e "Onset Encoder Model: ${EMFILE1}"
echo -e "Onset Decoder Model: ${DMFILE1}"
echo -e "Loss: ${LFILE}"

mkdir -p loss
mkdir -p model
mkdir -p ${MDIR}

for e in $(seq ${START_EPOCH} $((${END_EPOCH}-1)))
do
    for num in $(seq 1 82) #
    do
        python3 src/onoffset_resnet_DataAug.py -d1 ${DHEAD}${num}_${FEAT1} -a1 ${AHEAD1}${num} -em1 ${EMFILE1} \
        -dm1 ${DMFILE1} -p ${num} -e ${e} -l ${LR} \
        --hs1 ${HS1} --hl1 ${HL1} --window-size ${WS} --single-epoch ${SE} --bi1 ${BIDIR1} \
        --loss-record ${LFILE} --batch-size ${BATCH} --norm ${NORM} --feat1 ${FEAT_NUM1} \
        -emt1 ${TREMFILE1} -dmt1 ${TRDMFILE1}
    done
    
done