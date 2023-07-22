#!/usr/bin/env bash
TOTAL_UPDATES=6030    # Total number of training steps
PEAK_LR=0.001          # Peak learning rate, adjust as needed
WARMUP_UPDATES=482    # Warmup the learning rate over this many updates
MAX_TOKENS=16384        # Number of sequences per batch (batch size)
UPDATE_FREQ=64          # Increase the batch size 64x

for SEED in 4321; do 
SAVE_PATH=checkpoints/fine-tuned-BTR/
DATA_DIR=data-bin/clang8-conll13-conll14-t5tokenizer/

CUDA_VISIBLE_DEVICES=1 python fairseq/train.py $DATA_DIR \
    --seed $SEED \
    --update-ordered-indices-seed \
    --update-epoch-batch-itr True \
    --no-epoch-checkpoints \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task bidecoder_translation -s src -t tgt --criterion bidecoder_cross_entropy \
    --arch hf_T5 \
    --max-source-positions 128 \
    --max-target-positions 128 \
    --optimizer adafactor \
    --lr $PEAK_LR \
    --update-freq $UPDATE_FREQ \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --max-tokens-valid 128 \
    --bidirectional-decoder \
    --discriminative-size 20 \
    --train-cand-path data/TRAINING_clang8/ \
    --valid-cand-path data/DEV_CoNLL13_cleaned/ \
    --valid-m2 data/DEV_CoNLL13_cleaned/official-preprocessed-cleanpunc.m2 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 

done
