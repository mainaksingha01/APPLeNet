#!/bin/bash

cd ../..

# custom config
DATA=/raid/dgx1users/ankitj/Mainak/AppleNet/Data
TRAINER=AppleNet

DATASET=$1
SEED=$2

CFG=vit_b16_c4
SHOTS=16
SUB=all

#--load-epoch 50 \

DIR=Mainak/AppleNet/outputs/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "The results already exist in ${DIR}"
else
    python Mainak/AppleNet/train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file Mainak/AppleNet/yaml/datasets/${DATASET}.yaml \
    --config-file Mainak/AppleNet/yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir Mainak/AppleNet/outputs/crosstransfer/patternnet/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi