import os
from string import punctuation

from .util import is_module_available, get_module_or_attr

""" default paths """

DEFAULT_DATA_PATH = os.path.join(os.path.split(__file__)[0], "../data")
print(f"data folder is set to `{DEFAULT_DATA_PATH}` script")
if not os.path.exists(DEFAULT_DATA_PATH):
    os.makedirs(DEFAULT_DATA_PATH)

DEFAULT_TRAINTEST_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "traintest")

ALLENNLP_ELMO_PRETRAINED_FOLDER = os.path.join(DEFAULT_DATA_PATH, "allennlp_elmo_pretrained")

""" special tokenizers """

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
