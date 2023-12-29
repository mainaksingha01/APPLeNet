#!/bin/bash

cd ../..

# custom config
DATA=/Data
TRAINER=AppleNet

DATASET=$1
SEED=$2

CFG=vit_b16_c4
SHOTS=16
# LOADEP=30
SUB=new 

# --load-epoch ${LOADEP} \


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

MODEL_DIR=outputs/base2new/train_base/${COMMON_DIR}
DIR=outputs/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "The results already exist in ${DIR}"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi