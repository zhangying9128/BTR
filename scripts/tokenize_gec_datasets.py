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

    def encode_lines(self, line):
        source, target = line
        source = " ".join(self.model.tokenize(source))
        target = " ".join(self.model.tokenize(target))
        return ["PASS", [source, target]]

def process_texts(model, args):
    sources = []
    targets = []
    with open(os.path.join(args.path, ".".join([args.dataset, args.src])), "r") as f1, open(os.path.join(args.path, ".".join([args.dataset, args.tgt])), "r") as f2:
        for source, target in zip(f1.readlines(), f2.readlines()):
            sources.append(source)
            targets.append(target)

    encoder = MultiprocessingEdit()
    pool = Pool(60, initializer=encoder.initializer(model))

    time1=time.time()
    _sources = []
    _targets = []
    encoded_lines = pool.imap(encoder.encode_lines, zip(sources,targets))
    for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
        if filt == "PASS":
            if args.dataset == 'clang8':
                if len(enc_lines[0].split()) > 126 or len(enc_lines[1].split()) > 126:
                    continue
            _sources.append(enc_lines[0])
            _targets.append(enc_lines[1])

        if i % 10000 == 0:
            print("processed {} lines".format(i))

    time2=time.time()
    print('总共耗时：' + str(time2 - time1) + 's')

    with open(os.path.join(args.path, ".tok.".join([args.dataset, args.src])), "w") as f:
        for line in _sources:
            f.write(line + "\n")

    with open(os.path.join(args.path, ".tok.".join([args.dataset, args.tgt])), "w") as f:
        for line in _targets:
            f.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="dataset")
    parser.add_argument('--src', type=str, default="src")
    parser.add_argument('--tgt', type=str, default="tgt")
    parser.add_argument('--dataset', type=str, default="conll13", choices=["clang8", "conll14", "conll13", "bea", "jfleg"])
    args = parser.parse_args()


    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    process_texts(tokenizer, args)


