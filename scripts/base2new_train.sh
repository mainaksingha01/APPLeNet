#!/bin/bash

cd ../..

# custom config
DATA=/raid/dgx1users/ankitj/Mainak/AppleNet/Data
TRAINER=AppleNet

DATASET=$1
SEED=$2

CFG=vit_b16_c4 
SUB=base
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)


DIR=Mainak/AppleNet/outputs/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi