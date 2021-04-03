import os
from abc import ABC, abstractmethod
from string import punctuation
from typing import List

from .seq_modeling.util import is_module_available, get_module_or_attr

DEFAULT_DATA_PATH = os.path.join(os.path.split(__file__)[0], "../data")
print(f"data folder is set to `{DEFAULT_DATA_PATH}` script")
if not os.path.exists(DEFAULT_DATA_PATH):
    os.makedirs(DEFAULT_DATA_PATH)

ALLENNLP_ELMO_PRETRAINED_FOLDER = os.path.join(DEFAULT_DATA_PATH, "allennlp_elmo_pretrained")

ARXIV_CHECKPOINTS = {
    "bertscrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/bertscrnn-probwordnoise",
    "cnn-lstm-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/cnn-lstm-probwordnoise",
    "elmoscrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/elmoscrnn-probwordnoise",
    "lstm-lstm-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/lstm-lstm-probwordnoise",
    "scrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnn-probwordnoise",
    "scrnnbert-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnnbert-probwordnoise",
    "scrnnelmo-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnnelmo-probwordnoise",
    "subwordbert-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/subwordbert-probwordnoise",
}


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

    def finetune(self, *args, **kwargs):
        raise NotImplementedError

    def add_(self, contextual_model, at="input"):
        raise Exception("this functionality is only available with `SclstmChecker`")


def _is_punct(inp):
    return all([i in punctuation for i in inp])


if is_module_available("spacy"):
    spacy = get_module_or_attr("spacy")
    my_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    get_tokens = lambda inp: [token.text for token in my_nlp(inp)]
else:
    get_tokens = lambda inp: inp.split()


def custom_tokenizer(inp: str):
    tokens = get_tokens(inp)
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


spacy_tokenizer = custom_tokenizer
