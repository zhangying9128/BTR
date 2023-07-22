### cLang-8 dataset
Please follow the [steps](https://github.com/google-research-datasets/clang8) to process the cLang-8 dataset into a file clang8_source_target_en.spacy_tokenized.tsv.
And then split this file into clang8.src and clang8.tgt.

### CoNLL-13 dataset
Please download the [dataset](https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz) and split the file revised/data/official-preprocessed.m2 into files conll13.src and conll13.tgt

### CoNLL-14 dataset
Please download the [dataset](https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz) and split the file alt/official-2014.combined-withalt.m2 into files conll14.src and conll14.tgt

### BEA dataset
Please download the [dataset](https://competitions.codalab.org/competitions/20229) and use the file ABCN.test.bea19.orig as both bea.src and bea.tgt. (The .tgt file would not be utilized by the model, resulting in it can be anything. Here, we make files .src and .tgt to be same for convenience.) 

### JFLEG dataset
Please download the [dataset](https://github.com/keisks/jfleg) and use files test/test.src and test/test.ref0 as jfleg.src and jfleg.tgt, respectively.
