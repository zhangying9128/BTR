#!/usr/bin/env bash
#tokenize training or validation files
python scripts/tokenize_gec_datasets.py --path datasets/conll13/ --src conll13.src --tgt conll13.tgt --dataset conll13


#preprocessing for using fairseq
#you can set trainpref, validpref, or testpref according to your needs.
#for example, using jfleg datasets for testpref
python fairseq/preprocess.py \
    --source-lang src --target-lang tgt \
    --srcdict T5_model/dict.txt \
    --tgtdict T5_model/dict.txt \
    --destdir data-bin/clang8-conll13-conll14-t5tokenizer\
    --trainpref datasets/clang8/clang8.tok \
    --validpref datasets/conll13/conll13.tok \
    --testpref datasets/conll14/conll14.tok \
    --workers 60

