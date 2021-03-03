import os
from abc import ABC, abstractmethod
from string import punctuation
from typing import List

import spacy

""" constants """

DEFAULT_DATA_PATH = os.path.join(os.path.split(__file__)[0], "../data")
print(f"data folder is set to {DEFAULT_DATA_PATH}")
# assert os.path.isabs(DEFAULT_DATA_PATH)
# if not os.path.isdir(DEFAULT_DATA_PATH):
#     print("******")
#     print(f"data folder is set to {DEFAULT_DATA_PATH}. If incorrect, please replace it with correct path.")
#     print("******")

""" base class """


class Corrector(ABC):

    @abstractmethod
    def from_pretrained(self, ckpt_path, vocab="", weights=""):
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device='cpu'):
        raise NotImplementedError

    @abstractmethod
    def correct(self, x):
        raise NotImplementedError

    @abstractmethod
    def correct_string(self, mystring: str, return_all=False):
        raise NotImplementedError

    @abstractmethod
    def correct_strings(self, mystrings: List[str], return_all=False):
        raise NotImplementedError

    @abstractmethod
    def correct_from_file(self, src, dest="./clean_version.txt"):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, clean_file, corrupt_file):
        raise NotImplementedError

    @abstractmethod
    def model_size(self):
        raise NotImplementedError

    def finetune(self, clean_file="clean.txt", corrupt_file="corrupt.txt", new_vocab=None):

        if new_vocab is None:
            new_vocab = []
        if new_vocab:
            raise NotImplementedError("Do not currently support modifying output vocabulary of the models")

        raise NotImplementedError

    def add_(self, contextual_model, at="input"):

        raise Exception("this functionality is only available with `SclstmChecker`")


""" spacy tokenizer """


def _is_punct(inp):
    return all([i in punctuation for i in inp])


def spacy_tokenizer_tokens(inp):
    return [token.text for token in _spacy_nlp(inp)]


def custom_tokenizer(inp: str):
    tokens = spacy_tokenizer_tokens(inp)
    new_tokens = []
    str_ = ""
    for token in tokens:
        if _is_punct(token):
            str_ += token
        else:
            new_tokens.append(str_)
            str_ = ""
            new_tokens.append(token)
    if str_:
        new_tokens.append(str_)
    return " ".join(new_tokens)


_spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = custom_tokenizer
