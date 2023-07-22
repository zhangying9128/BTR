#!/usr/bin/env bash
# downloads training file
for FILE in {00000..00001} ; do 
SAVE_FILE=c4-train.$FILE-of-00512.json.gz
PULL_FILE=https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/$SAVE_FILE
wget -P datasets/realnewslike/ $PULL_FILE
gzip -d datasets/realnewslike/$SAVE_FILE
done

#downloads validation file
SAVE_FILE=c4-validation.00000-of-00001.json.gz
PULL_FILE=https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/$SAVE_FILE
wget -P datasets/realnewslike/ $PULL_FILE
gzip -d datasets/realnewslike/$SAVE_FILE

#tokenize training and validation files
python scripts/tokenize_realnewslike.py --realnewslike-path datasets/realnewslike/

#preprocessing for using fairseq
python fairseq/preprocess.py \
    --source-lang src --target-lang tgt \
    --srcdict T5_model/dict.txt \
    --tgtdict T5_model/dict.txt \
    --destdir data-bin/realnewslike-t5tokenizer \
    --trainpref datasets/realnewslike/train \
    --validpref datasets/realnewslike/valid \
    --workers 60
