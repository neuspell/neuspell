

"""
HOW TO RUN:
python run.py ./mnli_sentences_eng.retokenize False ./mnli_sentences_eng.retokenize ./mnli_sentences_eng.retokenize.noise.word
python run.py ../../data/traintest/train.1blm False ./large_files/train.1blm ../../data/traintest/train.1blm.noise.random
python run.py ../../data/traintest/test.1blm False ./large_files/test.1blm ../../data/traintest/test.1blm.noise.random
"""


import os, sys
from tqdm import tqdm


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
# import en_core_web_sm
_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]


from utils import get_line_representation

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

    print(len(inp))

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
    print("total lines in retokenized_lines: {}".format(len(retokenized_lines)))
    print("total tokens in retokenized_lines: {}".format(sum([len(line.strip().split()) for line in retokenized_lines])))

    new_lines = get_line_representation(retokenized_lines)
    assert len(new_lines)==len(retokenized_lines)

    opfile = open(CLEAN_FILE_PATH_,"w")
    for line in tqdm(retokenized_lines[:-1]):
        opfile.write(line+"\n")
    opfile.write(retokenized_lines[-1])
    opfile.close()

    opfile = open(NOISE_FILE_PATH_,"w")
    for line in tqdm(new_lines[:-1]):
        opfile.write(line+"\n")
    opfile.write(new_lines[-1])
    opfile.close()

