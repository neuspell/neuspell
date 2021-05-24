from string import punctuation

from .util import is_module_available, get_module_or_attr

_SPACY_TOKENIZER, _SPACY_TAGGER = None, None


def _load_spacy_tokenizer(model_name):
    global _SPACY_TOKENIZER

    if not _SPACY_TOKENIZER:
        if is_module_available("spacy"):
            if not is_module_available(model_name):
                raise ImportError(f"run `python -m spacy download {model_name}`")
            spacy_nlp = get_module_or_attr(model_name).load(disable=["tagger", "ner", "lemmatizer"])
            _SPACY_TOKENIZER = lambda inp: [token.text for token in spacy_nlp(inp)]
            print(f"spacy tokenizer from model {model_name} initialized")
        else:
            raise ImportError("`pip install spacy` to use spacy retokenizer")
    return _SPACY_TOKENIZER


def _load_spacy_tagger(model_name):
    global _SPACY_TAGGER

    if not _SPACY_TAGGER:
        if is_module_available("spacy"):
            if not is_module_available(model_name):
                raise ImportError(f"run `python -m spacy download {model_name}`")
            spacy_nlp = get_module_or_attr(model_name).load(disable=["ner", "lemmatizer"])
            _SPACY_TAGGER = lambda inp: [token.tag for token in spacy_nlp(inp)]
            print(f"spacy pos tagger from model {model_name} initialized")
        else:
            raise ImportError("`pip install spacy` to use spacy retokenizer")
    return _SPACY_TAGGER


def default_tokenizer(inp: str, model_name="en_core_web_sm"):
    try:
        _spacy_tokenizer = _load_spacy_tokenizer(model_name)
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
