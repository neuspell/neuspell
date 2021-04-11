import os
from abc import ABC, abstractmethod

from tqdm.autonotebook import tqdm

from ._util import _load_assorted_mistakes, _load_assorted_mistakes_mappings
from ._util import is_module_available, get_module_or_attr
from ._util import noisyfy_word_tokens
from .downloads import download_file_from_google_drive
from ..commons import DEFAULT_DATA_PATH, _load_spacy_tokenizer

DEFAULT_NOISING_RESOURCES_PATH = os.path.join(DEFAULT_DATA_PATH, "noising_resources")


class Noiser(ABC):

    @staticmethod
    def __spacy_retokenizer(texts):
        _spacy_tokenizer = _load_spacy_tokenizer()
        texts = [_x.strip() for _x in texts]
        retokenized_texts = []
        pbar = tqdm(total=1)
        i, bsz = 0, 5000
        while i >= 0:
            lines = " UNIQUE_SPLITTER ".join([line.strip() for line in texts[i:i + bsz]])
            tokens = _spacy_tokenizer(lines)
            lines = " ".join(tokens).split("UNIQUE_SPLITTER")
            lines = [line.strip() for line in lines]
            retokenized_texts += lines
            i += bsz
            pbar.update(bsz / len(texts))
            if i > len(texts): i = -1
        pbar.close()
        assert len(retokenized_texts) == len(texts), print(len(retokenized_texts), len(texts))
        return retokenized_texts

    @staticmethod
    def create_preprocessor(lower_case=False, remove_accents=False):

        # checks
        if remove_accents:
            assert is_module_available("unidecode"), print("pip install unidecode")

        # local methods
        _lower_case = lambda x: x.lower() if lower_case else x
        _remove_accents = lambda x: get_module_or_attr("unidecode", "unidecode")(x) if remove_accents else x

        # cummulative
        _preprocessor = lambda x: [_lower_case(_remove_accents(_x)) for _x in x]
        return _preprocessor

    @staticmethod
    def create_retokenizer(use_spacy_retokenization=False):
        if use_spacy_retokenization:
            _retokenizer = Noiser.__spacy_retokenizer
        else:
            _retokenizer = lambda x: [_x.strip() for _x in x]
        return _retokenizer

    @abstractmethod
    def load_resources(self):
        raise NotImplementedError

    @abstractmethod
    def _noise(self, texts, **kwargs):
        raise NotImplementedError

    def noise(self, texts, preprocessor=None, retokenizer=None, **kwargs):
        if not preprocessor:
            preprocessor = self.create_preprocessor()
        texts = preprocessor(texts)
        if not retokenizer:
            retokenizer = self.create_retokenizer()
        texts = retokenizer(texts)
        print(f"total # of texts after retokenization: {len(texts)}")
        print(f"total # of tokens after retokenization: {sum([len(line.strip().split()) for line in texts])}")
        return self._noise(texts, **kwargs)


class WordReplacementNoiser(Noiser):

    def __init__(self, language="english"):
        self.language = language.lower()
        self.resource_folder = os.path.join(DEFAULT_NOISING_RESOURCES_PATH, "en-word-replacement-noise")

        if not self.language == "english":
            raise ValueError("WordReplacementNoiser currently support only English language! "
                             "Set `language=english` in arguments")

        self.ready = False

    def load_resources(self):
        if not os.path.exists(self.resource_folder):
            os.makedirs(self.resource_folder)

        mistakes_vocab_path = os.path.join(self.resource_folder, "combined_data_homophones_stats.tsv")
        if not os.path.exists(mistakes_vocab_path):
            print(f"downloading resources in WordReplacementNoiser to folder {self.resource_folder}")
            download_file_from_google_drive("1Nr4TucTveelIDGyc894EHE6oLFsqaJe-", mistakes_vocab_path)
        else:
            print(f"Utilizing resources existing in folder {self.resource_folder}")
        self.mistakes_vocab = _load_assorted_mistakes(mistakes_vocab_path)

        mistakes_mappings_path = os.path.join(self.resource_folder, "combined_data_homophones.tsv")
        if not os.path.exists(mistakes_mappings_path):
            print(f"downloading resources in WordReplacementNoiser to folder {self.resource_folder}")
            download_file_from_google_drive("1ARfPoM6cwUicGV-qjlOMDXXAu6hY0Izl", mistakes_mappings_path)
        else:
            print(f"Utilizing resources existing in folder {self.resource_folder}")
        self.mistakes_mappings = _load_assorted_mistakes_mappings(mistakes_mappings_path)

        self.ready = True

        return

    def _noise(self, texts, **kwargs):
        """

        :param texts: list of texts to be noised
        :param kwargs:
        :return:
        """
        expected_prob = kwargs.get("expected_prob", 0.20)
        min_len = kwargs.get("min_len", 1)

        if not self.ready:
            raise Exception(f"Must call .load_resources() before using noising methods.")

        new_texts = noisyfy_word_tokens(texts, self.mistakes_vocab, self.mistakes_mappings,
                                        expected_prob=expected_prob, min_len=min_len)
        assert len(new_texts) == len(texts)

        return new_texts


class CharacterReplacementNoiser(Noiser):

    def __init__(self, language="english"):
        self.language = language.lower()
        if not self.language == "english":
            raise ValueError("CharacterReplacementNoiser currently support only English language! "
                             "Set `language=english` in arguments")

    def load_resources(self):
        raise NotImplementedError

    def _noise(self, texts, **kwargs):
        raise NotImplementedError


class ProbabilisticCharacterReplacementNoiser(Noiser):

    def __init__(self, language="english"):
        self.language = language.lower()
        if not self.language == "english":
            raise ValueError("ProbabilisticCharacterReplacementNoiser currently support only English language! "
                             "Set `language=english` in arguments")

    def load_resources(self):
        raise NotImplementedError

    def _noise(self, texts, **kwargs):
        raise NotImplementedError
