import os
from typing import List

import torch

from .commons import spacy_tokenizer, ARXIV_CHECKPOINTS, Corrector
from .seq_modeling.downloads import download_pretrained_model
from .seq_modeling.helpers import load_data, load_vocab_dict, get_model_nparams
from .seq_modeling.lstmlstm import load_model, load_pretrained, model_predictions, model_inference

""" corrector module """


class CorrectorLstmLstm(Corrector):

    def __init__(self, tokenize=True, pretrained=False, device="cpu"):
        super(CorrectorLstmLstm, self).__init__()
        self.tokenize = tokenize
        self.pretrained = pretrained
        self.device = device

        self.ckpt_path = None
        self.vocab_path, self.weights_path = "", ""
        self.model, self.vocab = None, None

        if self.pretrained:
            self.from_pretrained(self.ckpt_path)

    def __model_status(self):
        assert not (self.model is None or self.vocab is None), print("model & vocab must be loaded first")
        return

    def from_pretrained(self, ckpt_path=None, vocab="", weights=""):
        self.ckpt_path = ckpt_path or ARXIV_CHECKPOINTS["lstm-lstm-probwordnoise"]
        self.vocab_path = vocab if vocab else os.path.join(self.ckpt_path, "vocab.pkl")
        if not os.path.isfile(self.vocab_path):  # leads to "FileNotFoundError"
            download_pretrained_model(self.ckpt_path)
        print(f"loading vocab from path:{self.vocab_path}")
        self.vocab = load_vocab_dict(self.vocab_path)
        print(f"initializing model")
        self.model = load_model(self.vocab)
        self.weights_path = weights if weights else self.ckpt_path
        print(f"loading pretrained weights from path:{self.weights_path}")
        self.model = load_pretrained(self.model, self.weights_path, device=self.device)
        return

    def set_device(self, device='cpu'):
        prev_device = self.device
        device = "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        if not (prev_device == device):
            if self.model is not None:
                # please load again, facing issues with just .to(new_device) and new_device
                #   not same the old device, https://tinyurl.com/y57pcjvd
                self.from_pretrained(self.ckpt_path, vocab=self.vocab_path, weights=self.weights_path)
            self.device = device
        print(f"model set to work on {device}")
        return

    def correct(self, x):
        return self.correct_string(x)

    def correct_string(self, mystring: str, return_all=False) -> str:
        x = self.correct_strings([mystring], return_all=return_all)
        if return_all:
            return x[0][0], x[1][0]
        else:
            return x[0]

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.__model_status()
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 16
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def correct_from_file(self, src, dest="./clean_version.txt"):
        """
        src = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        self.__model_status()
        x = [line.strip() for line in open(src, 'r')]
        y = self.correct_strings(x)
        print(f"saving results at: {dest}")
        opfile = open(dest, 'w')
        for line in y:
            opfile.write(line + "\n")
        opfile.close()
        return

    def evaluate(self, clean_file, corrupt_file):
        """
        clean_file = f"{DEFAULT_DATA_PATH}/traintest/clean.txt"
        corrupt_file = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        self.__model_status()
        batch_size = 4 if self.device == "cpu" else 16
        for x, y, z in zip([""], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            _ = model_inference(self.model,
                                test_data,
                                topk=1,
                                device=self.device,
                                batch_size=batch_size,
                                vocab_=self.vocab)
        return

    def model_size(self):
        self.__model_status()
        return get_model_nparams(self.model)
