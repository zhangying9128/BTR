# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:48:29 2020

@author: zhang
"""
import os
import json
import argparse
from multiprocessing import Pool
import time

class MultiprocessingEdit(object):

    def __init__(self):
        pass

    def initializer(self, model):
        self.model = model

    def BPE_tokenize(self, sentence):
        bpe_sentence = self.model(sentence).input_ids[:-1]
        return bpe_sentence

    def encode_lines(self, line): 
        cand_edus = line[0]
        batch_edus = []
        for cand in cand_edus["gec_outputs"]:
            cand = self.BPE_tokenize(cand[0])
            batch_edus.append(cand)

        cand_edus["gec_outputs_binary"] = batch_edus
        if 'source' in cand_edus:
            cand_edus['source_strings_binary'] = self.BPE_tokenize(cand_edus['source'][0])
        return ["PASS", [cand_edus]]

def preprocess_texts(model, args):
    with open(os.path.join(args.path, args.input), "r") as f:
        inputs = [json.loads(line) for line in f.readlines()]

    encoder = MultiprocessingEdit()
    pool = Pool(60, initializer=encoder.initializer(model))

    time1 = time.time()
    outputs = []
    encoded_lines = pool.imap(encoder.encode_lines, zip(inputs))
    for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
        if filt == "PASS":
            outputs += enc_lines
        if i % 10000 == 0:
            print("processed {} lines".format(i))
    time2 = time.time()
    print('time costs:' + str(time2 - time1) + 's')

    with open(os.path.join(args.path, args.output), "w") as f:
        for line in outputs:
            f.write(json.dumps(line) + "\n")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="data/DEV_CoNLL13")
    parser.add_argument('--input', type=str, default="output.txt")
    parser.add_argument('--output', type=str, default="output.tok.txt")
    args = parser.parse_args()


    from transformers import T5Tokenizer
    model = T5Tokenizer.from_pretrained("t5-base")
    preprocess_texts(model, args)
