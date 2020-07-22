

"""
HOW TO RUN:
python run.py ./mnli_sentences_eng.retokenize False ./mnli_sentences_eng.retokenize ./mnli_sentences_eng.retokenize.noise.word
python run.py /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.1blm False ./delete /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.1blm.noise.word
# python run.py /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.1blm False ./delete /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.1blm.noise.ambiguous
# python run.py /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.bea60k False /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.bea60k.lowercased /projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.bea60k.lowercased.noise.ambiguous False
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
import en_core_web_sm
_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]


from utils import _load_assorted_mistakes, _load_assorted_mistakes_mappings, noisyfy_word_tokens

PATH1 = "./files/online_collections/combined_data_homophones_stats.tsv" #"combined_data_stats.tsv"
PATH2 = "./files/online_collections/combined_data_homophones.tsv" #"combined_data.tsv"
# PATH1 = "./files/ambiguous/ambiguous_data_stats.tsv"
# PATH2 = "./files/ambiguous/ambiguous_data.tsv"

if __name__=="__main__":

    CLEAN_FILE_PATH = sys.argv[1] #str file path
    RE_TOKENIZE = sys.argv[2] #str true/false
    CLEAN_FILE_PATH_ = sys.argv[3] #str file path
    NOISE_FILE_PATH_ = sys.argv[4] #str file path
    if len(sys.argv)>5:
        LOWER_CASE = sys.argv[5] #str true/false
    else:
        LOWER_CASE = False

    x = "/".join(CLEAN_FILE_PATH_.split("/")[:-1])
    if not os.path.exists(x): os.makedirs(x)
    x = "/".join(NOISE_FILE_PATH_.split("/")[:-1])
    if not os.path.exists(x): os.makedirs(x)

    opfile = open(CLEAN_FILE_PATH,"r")
    inp = opfile.readlines()
    opfile.close()
    print("total lines in inp: {}".format(len(inp)))
    print("total tokens in inp: {}".format(sum([len(line.strip().split()) for line in inp])))

    # print(len(inp))

    if LOWER_CASE:
            print("LOWER_CASE...")
            inp = [line.lower() for line in inp]

    print(RE_TOKENIZE)
    if RE_TOKENIZE.lower()=="true":
        print("RE_TOKENIZE...")
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

    mistakes_vocab = _load_assorted_mistakes(PATH1) 
    mistakes_mappings = _load_assorted_mistakes_mappings(PATH2) 

    new_lines = noisyfy_word_tokens(retokenized_lines, mistakes_vocab, mistakes_mappings, expected_prob=0.20)
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


"""
outputs:
# unique tokens overlapped: 14815
# total tokens overlapped: 19581925
# overlap percent wrt original_sentences: 64.3485
# overlap percent wrt mistakes_vocab: 81.7019
# #overlap_count:19581925, #total_count:30431058, #overlap_percent:64.34848568196347
# ------------------------------------
# 31.0808% of overlapped tokens will get replaced to match the total % of misspellings to 20.0000%
# Percentage of tokens that actually got replaced 6081831/30431058=19.9856%
# No of tokens in mistakes_mappings queried: 14051
"""