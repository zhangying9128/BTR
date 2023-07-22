import os
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import time

class MultiprocessingEdit(object):

    def __init__(self):
        pass

    def initializer(self, model):
        self.model = model

    def encode_lines(self, line):
        source = line[0]
        source = " ".join(self.model.tokenize(source))
        target = line[1]
        target = " ".join(self.model.tokenize(target))
        return ["PASS", [source, target]]

def preprocess_train(model, args):
    encoder = MultiprocessingEdit()
    pool = Pool(60, initializer=encoder.initializer(model))
    for i in range(512):
        s = str('{:0>5d}'.format(i))
        train_file = 'c4-train.' + s + '-of-00512.json'

        sources = []
        targets = [] 
        with open(os.path.join(args.realnewslike_path, train_file)) as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                for j, sentence in enumerate(text.split('\n')):
                    sources.append(sentence)
                    targets.append('target')

        time1=time.time()
        _sources = []
        _targets = []
        encoded_lines = pool.imap(encoder.encode_lines, zip(sources, targets))
        for j, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                _sources.append(enc_lines[0])
                _targets.append(enc_lines[1])
            if j % 10000 == 0:
                print("processed {} lines".format(j))

        time2=time.time()
        print('time cost:' + str(time2 - time1) + 's')

        #save subfiles in case error happens
        with open(os.path.join(args.realnewslike_path, "train."+str(i)+".src"), "w") as f:
            for line in _sources:
                f.write(line + "\n")

        with open(os.path.join(args.realnewslike_path, "train."+str(i)+".tgt"), "w") as f:
            for line in _targets:
                f.write(line + "\n")

def preprocess_valid(model, args):
    sources = []
    targets = []

    with open(args.realnewslike_path + 'c4-validation.00000-of-00001.json') as f:
        for line in f.readlines():
            text = json.loads(line)['text']
            for i, sentence in enumerate(text.split('\n')):
                sources.append(sentence)
                targets.append('target')


    encoder = MultiprocessingEdit()
    pool = Pool(60, initializer=encoder.initializer(model))

    time1=time.time()
    _sources = []
    _targets = []
    encoded_lines = pool.imap(encoder.encode_lines, zip(sources,targets))
    for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
        if filt == "PASS":
            _sources.append(enc_lines[0])
            _targets.append(enc_lines[1])

    time2=time.time()
    print('time cost:' + str(time2 - time1) + 's')

    with open(os.path.join(args.realnewslike_path, "valid.src"), "w") as f:
        for line in _sources:
            f.write(line + "\n")

    with open(os.path.join(args.realnewslike_path, "valid.tgt"), "w") as f:
        for line in _targets:
            f.write(line + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--realnewslike-path', type=str, default="realnewslike")
    args = parser.parse_args()

    #check tokenizer
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    input_sentece = 'The <extra_id_0> walks in <extra_id_1> park'
    input_ids = tokenizer(input_sentece)
    input_ids = tokenizer.tokenize(input_sentece)
    print('----------check tokenizer---------')
    print(input_sentece)
    print(input_ids)
    print(tokenizer.convert_tokens_to_ids(input_ids))


    preprocess_valid(tokenizer, args)
    preprocess_train(tokenizer, args)


    for item in [".src", ".tgt"]:
        sentences = []
        for i in tqdm(range(512)):
            with open(os.path.join(args.realnewslike_path, "train."+str(i)+ item)) as f1:
                sentences += f1.readlines()

        with open(os.path.join(args.realnewslike_path, "train" + item), "w") as f1:
            for sentence in sentences:
                f1.write(sentence)

