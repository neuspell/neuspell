import os

""" spacy tokenization """

from string import punctuation
is_punct = lambda inp: all([i in punctuation for i in inp])

import spacy
spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer_tokens = lambda inp: [token.text for token in spacy_nlp(inp)]
def custom_tokenizer(inp: str):
    tokens = spacy_tokenizer_tokens(inp)
    new_tokens = []
    str_ = ""
    for token in tokens:
        if is_punct(token):
            str_+=token
        else:
            new_tokens.append(str_)
            str_ = ""
            new_tokens.append(token)
    if str_:
        new_tokens.append(str_)
    return " ".join(new_tokens)
spacy_tokenizer = custom_tokenizer

""" constants """
DEFAULT_DATA_PATH = "/neuspell/data/"
assert os.path.isabs(DEFAULT_DATA_PATH)
if not os.path.isdir(DEFAULT_DATA_PATH):
    print("******")
    print(f"data folder is set to {DEFAULT_DATA_PATH}. If incorrect, please replace it with correct absoulte path.")
    print("******")
