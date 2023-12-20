#!/usr/bin/env bash
CHECKPOINT=checkpoints/fine-tuned-BTR/checkpoint_last.pt
DATA_DIR=data-bin/clang8-conll13-conll14-t5tokenizer/
OUTPUT=checkpoints/scores.txt

CUDA_VISIBLE_DEVICES=1 python fairseq/validate.py $DATA_DIR \
    --task bidecoder_translation -s src -t tgt \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --path $CHECKPOINT \
    --results-path $OUTPUT \
    --num-of-candidates 5 \
    --max-tokens-valid 512 \
    --bidirectional-decoder \
    --test-cand-path data/TEST_CoNLL14/ \
    --log-format simple --log-interval 60 
