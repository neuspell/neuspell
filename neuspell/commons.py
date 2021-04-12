import os
from abc import ABC, abstractmethod
from string import punctuation
from typing import List

from .seq_modeling.util import is_module_available, get_module_or_attr

DEFAULT_DATA_PATH = os.path.join(os.path.split(__file__)[0], "../data")
print(f"data folder is set to `{DEFAULT_DATA_PATH}` script")
if not os.path.exists(DEFAULT_DATA_PATH):
    os.makedirs(DEFAULT_DATA_PATH)

DEFAULT_TRAINTEST_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "traintest")

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


_SPACY_TOKENIZER, _SPACY_TAGGER = None, None


def _load_spacy_tokenizer():
    global _SPACY_TOKENIZER, _SPACY_TAGGER

    if not _SPACY_TOKENIZER:
        if is_module_available("spacy"):
            if not is_module_available("en_core_web_sm"):
                raise ImportError("run `python -m spacy download en_core_web_sm`")
            print("creating spacy models ...")
            spacy_nlp = get_module_or_attr("en_core_web_sm").load(disable=["tagger", "ner", "lemmatizer"])
            _SPACY_TOKENIZER = lambda inp: [token.text for token in spacy_nlp(inp)]
            # spacy_nlp = get_module_or_attr("en_core_web_sm").load(disable=["ner", "lemmatizer"])
            # _SPACY_TAGGER = lambda inp: [token.tag for token in spacy_nlp(inp)]
            print("spacy models initialized")
        else:
            raise ImportError("`pip install spacy` to use spacy retokenizer")
    return _SPACY_TOKENIZER


def _custom_tokenizer(inp: str):
    try:
        _spacy_tokenizer = _load_spacy_tokenizer()
        get_tokens = lambda inp: _spacy_tokenizer(inp)
    except ImportError as e:
        print(e)
        get_tokens = lambda inp: inp.split()

    def _is_punct(inp):
        return all([i in punctuation for i in inp])

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


spacy_tokenizer = _custom_tokenizer
