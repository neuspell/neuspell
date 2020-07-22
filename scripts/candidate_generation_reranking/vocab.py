
"""
for creating vocab.txt, use as follows:
-------
from utils.vocab import Vocab
paths = ["../data/vocab/transformerxl.txt","../data/vocab/phonemedataset.txt","../data/vocab/tatoeba.txt"]
vocab = Vocab(paths)
vocab.save_vocab("../data/vocab/vocab.txt")


for loading vocab.txt, use as follows:
-------
from utils.vocab import Vocab
paths = ["../data/vocab/vocab.txt"]
vocab = Vocab(paths)
"""

import numpy as np
import os
import torch
from tqdm import tqdm
import string


class Vocab(object):

    def __init__(self, lexicon_paths=[]):
        
        self.only_ascii = True
        self.without_numbers = True
        self.without_puncts = True

        self.token2idx = {}
        self.idx2token = {}
        self.size = 0        

        for path_ in lexicon_paths:
            self.load_tokens(path_)

    def add_tokens(self, tokens):
        isascii = lambda s: len(s) == len(s.encode()) if self.only_ascii else True
        isnumbered = lambda s: len([x for x in list(s) if x.isdigit()])>0 if self.without_numbers else False
        punct_list = string.punctuation.replace("'","").replace("-","").replace("@","")
        ispuncted = lambda s: len([x for x in s if x in punct_list])>0 if self.without_puncts else True

        tokens = [token for token in tokens if (isascii(token) and not isnumbered(token) and not ispuncted(token))]
        for token in tqdm(tokens):
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        self.size = len(self.token2idx)
        return

    def load_tokens(self, file_path):
        tokens = []
        opfile = open(file_path,"r")
        tokens = [line.split("\t")[0].strip() for line in opfile]
        tokens = [token for token in tokens if token!=""]
        opfile.close()
        self.add_tokens(tokens)
        return

    def save_vocab(self, file_path):
        x = "/".join(file_path.split("/")[:-1])
        if not os.path.exists(x):
            os.makedirs(x)
        opfile = open(file_path,"w")
        for k,v in self.token2idx.items():
            opfile.write(f"{k}\t{v}\n")
        opfile.close()
        print(file_path)
        return




"""
pip install transformers
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize as nltk_tokenizer
pip install -U spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python -m spacy download en_core_web_sm
"""


"""
opfile = open("vocab_ascii.txt","w")
for token in vocablist:
    if token:
        opfile.write(token+"\n")
opfile.close()
"""

"""
print(spacy_tokenizer("Two adults 100$, wow!!! kdnckn :p walk across the street to get away from a red shirted person who is chasing them."))
print(nltk_tokenizer("Two adults 100$, wow!!! kdnckn :p walk across the street to get away from a red shirted person who is chasing them."))
print(nltk.tokenize.TreebankWordTokenizer().tokenize("Two adults 100$, wow!!! kdnckn :p walk across the street to get away from a red shirted person who is chasing them."))
print(spacy_tokenizer("P. S.--I wish you would measure one of the largest of those swords we took to Alton and write me the length of it, from tip of the point to tip of the hilt, in feet and inches. I have a dispute about the length. A. L. A. L."))
print(nltk_tokenizer("P. S.--I wish you would measure one of the largest of those swords we took to Alton and write me the length of it, from tip of the point to tip of the hilt, in feet and inches. I have a dispute about the length. A. L. A. L."))
print(nltk.tokenize.TreebankWordTokenizer().tokenize("P. S.--I wish you would measure one of the largest of those swords we took to Alton and write me the length of it, from tip of the point to tip of the hilt, in feet and inches. I have a dispute about the length. A. L. A. L."))
print(spacy_tokenizer("\"This is what I was looking for\", he explained."))
print(nltk_tokenizer("\"This is what I was looking for\", he explained."))
print(spacy_tokenizer("Two adults 100$, wow!!! kdnckn :p walk across the street UNIQUE_SPLITTER to get away from a red shirted person who is chasing them."))
print(nltk_tokenizer("Two adults 100$, wow!!! kdnckn :p walk across the street UNIQUE_SPLITTER to get away from a red shirted person who is chasing them."))
print(spacy_tokenizer("I wonder/how this expression!!!?? gets spl-it into...!"))
print(nltk_tokenizer("I wonder/how this expression!!!?? gets spl-it into...!"))
print(spacy_tokenizer("A(n) polyploid is an individual with mo."))
print(nltk_tokenizer("A(n) polyploid is an individual with mo."))
print(spacy_tokenizer("the night-walker guy is amazing"))
print(nltk_tokenizer("the night-walker guy is amazing"))
print(spacy_tokenizer("\"This is what I was looking for\", he explained."))
print(nltk_tokenizer("\"This is what I was looking for\", he explained."))
print(spacy_tokenizer("P. S.--I wish you would measure one of the largest of those swords we took to Alton and write me the length of it, from tip of the point to tip of the hilt, in feet and inches. I have a dispute about the length. A. L. A. L."))
print(nltk_tokenizer("P. S.--I wish you would measure one of the largest of those swords we took to Alton and write me the length of it, from tip of the point to tip of the hilt, in feet and inches. I have a dispute about the length. A. L. A. L."))
"""

"""
import spacy
import en_core_web_sm
_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]

def _set_tatoeba_vocab(self):
    from downloads import download_tateoba_english_sentences
    write_path = download_tateoba_english_sentences("./temp")
    opfile = open(write_path,"r")
    inp = opfile.readlines()
    opfile.close()
    print(len(inp))

    isascii = lambda s: len(s) == len(s.encode())
    def add_vocab(word, d): 
        if not word in d: d[word]=0
        d[word]+=1
        return d
    vocab = {}
    pbar = tqdm(total=1)
    i, bsz = 0, 10000
    while i>=0:
        lines = " ".join([line.strip() for line in inp[i:i+bsz]])
        tokens = spacy_tokenizer(lines)
        for token in tokens:
            if token:
                vocab = add_vocab(token, vocab)
        i+=bsz
        pbar.update(bsz/len(inp))
        if i>len(inp): i=-1
    pbar.close()
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}
    tokens = [*vocab.keys()]
    self.add_tokens(tokens)
    return tokens

def _set_transformerxl_vocab(self):
    from transformers import TransfoXLModel, TransfoXLTokenizer
    model_class = TransfoXLModel
    tokenizer_class = TransfoXLTokenizer
    pretrained_weights = 'transfo-xl-wt103'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    BASE_PATH = './temp/transformerxl'
    MODEL_SAVE_PATH = os.path.join(BASE_PATH,pretrained_weights)
    if not(os.path.exists(MODEL_SAVE_PATH)):
        os.makedirs(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    vocab_dict = torch.load(os.path.join(BASE_PATH,"transfo-xl-wt103/vocab.bin"))
    vocab_txt = os.path.join(BASE_PATH,"vocab.txt")
    opfile = open(vocab_txt,"w")
    for token in vocab_dict['idx2sym']:
        opfile.write("{}\n".format(token))
    opfile.close()
    self.add_tokens(vocab_dict['idx2sym'])
    return

def _set_bert_vocab(self):
    from transformers import BertForMaskedLM, BertTokenizer
    model_class = BertForMaskedLM
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    BASE_PATH = './temp/bert'
    MODEL_SAVE_PATH = os.path.join(BASE_PATH,pretrained_weights)
    if not(os.path.exists(MODEL_SAVE_PATH)):
        os.makedirs(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    VOCAB_FILE = os.path.join(MODEL_SAVE_PATH, 'vocab.txt')
    opfile = open(vocab_txt,"r")
    tokens = [token.strip() for token in opfile]
    opfile.close()
    self.add_tokens(tokens)
    return

def _merge_vocab(self, vocabs: "list of objects; type Vocab()"):
    tokens = []
    for vocab in vocabs: tokens += [*vocab.token2idx.keys()]
    self.add_tokens(tokens, only_ascii)
    return
"""