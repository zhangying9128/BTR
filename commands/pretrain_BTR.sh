#!/usr/bin/env bash
TOTAL_UPDATES=65536    # Total number of training steps 
PEAK_LR=0.001          # Peak learning rate, adjust as needed
MAX_TOKENS=16384        # Number of sequences per batch (batch size) 16384
UPDATE_FREQ=32          # Increase the batch size 32x

for SEED in 4321; do 
DATA_DIR=data-bin/realnewslike-t5tokenizer/
SAVE_PATH=checkpoints/pre-trained-BTR/

CUDA_VISIBLE_DEVICES=0,1 python fairseq/train.py $DATA_DIR \
    --seed $SEED \
    --update-ordered-indices-seed \
    --update-epoch-batch-itr True \
    --no-epoch-checkpoints \
    --distributed-world-size 2 \
    --ddp-backend legacy_ddp \
    --task bidecoder_translation -s src -t tgt --criterion bidecoder_cross_entropy \
    --arch hf_T5 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --optimizer adafactor \
    --lr $PEAK_LR \
    --update-freq $UPDATE_FREQ \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --max-tokens-valid 8192 \
    --bidirectional-decoder \
    --bidirectional-pretrain \
    --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1

done

