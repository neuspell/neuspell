


"""
HOW TO RUN:
python run.py ./mnli_sentences_eng.txt True ./mnli_sentences_eng.retokenize ./mnli_sentences_eng.retokenize.noise.prob
"""


import numpy as np
np.random.seed(11690)
import os
from tqdm import tqdm
import sys

"""
import nltk
nltk.download('punkt')
#from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize as nltk_tokenizer
"""
"""
try: import spacy
except: 
    os.system("pip install -U spacy"); 
    os.system("pip install \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz")
    #os.system("!python -m spacy download en_core_web_sm")
"""
import spacy
import en_core_web_sm
_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]

from utils import noisyfy_backoff_homophones
from stats import load_stats, get_stats, save_stats, to_probs

STATS_JSON_PATH = './large_files/moe_misspellings_train_ascii_stats_left_context.json'
HOMOPHONES_PATH = './large_files/homophones.txt'

if __name__=="__main__":

    CLEAN_FILE_PATH = sys.argv[1] #str file path
    RE_TOKENIZE = sys.argv[2] #str true/false
    CLEAN_FILE_PATH_ = sys.argv[3] #str file path
    NOISE_FILE_PATH_ = sys.argv[4] #str file path

    x = "/".join(CLEAN_FILE_PATH_.split("/")[:-1])
    if not os.path.exists(x): os.makedirs(x)
    x = "/".join(NOISE_FILE_PATH_.split("/")[:-1])
    if not os.path.exists(x): os.makedirs(x)

    opfile = open(CLEAN_FILE_PATH,"r")
    inp = opfile.readlines()
    opfile.close()
    print("total lines in inp: {}".format(len(inp)))
    print("total tokens in inp: {}".format(sum([len(line.strip().split()) for line in inp])))

    print(RE_TOKENIZE)
    if RE_TOKENIZE.lower()=="true":
        retokenized_lines = []
        pbar = tqdm(total=1)
        i, bsz = 0, 5000
        while i>=0:
            lines = " UNIQUE_SPLITTER ".join([line.strip() for line in inp[i:i+bsz]])
            tokens = spacy_tokenizer(lines)
            lines = " ".join(tokens).split("UNIQUE_SPLITTER")
            lines = [line.strip() for line in lines]
            retokenized_lines += lines
            i+=bsz
            pbar.update(bsz/len(inp))
            if i>len(inp): i=-1
        pbar.close()
        assert len(retokenized_lines) == len(inp)
    else:
        retokenized_lines = [line.strip() for line in inp]
    print("total lines in retokenized_inp: {}".format(len(retokenized_lines)))
    print("total tokens in retokenized_inp: {}".format(sum([len(line.strip().split()) for line in retokenized_lines])))

    existing_stats = load_stats(STATS_JSON_PATH)
    stats = existing_stats
    stats = to_probs(stats) 

    homophones = {}
    opfile = open(HOMOPHONES_PATH,'r')
    for line in opfile:
        w1, w2 = line.strip().split('\t')
        try:
            homophones[w1].append(w2)
        except KeyError:
            homophones[w1] = [w2]
    opfile.close()
    homophones_set = set([*homophones.keys()])

    opfile = open(CLEAN_FILE_PATH_,"w")
    for line in tqdm(retokenized_lines[:-1]):
        opfile.write(line+"\n")
    opfile.write(retokenized_lines[-1])
    opfile.close()

    new_lines = noisyfy_backoff_homophones(stats,retokenized_lines,[0.025, 0.05, 0.2, 0.7],homophones,0)
    assert len(new_lines)==len(retokenized_lines)
    opfile = open(NOISE_FILE_PATH_,"w")
    for line in tqdm(new_lines[:-1]):
        opfile.write(line+"\n")
    opfile.write(new_lines[-1])
    opfile.close()

