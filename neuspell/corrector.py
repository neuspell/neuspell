import os
from abc import ABC, abstractmethod
from typing import List

import torch

from .commons import DEFAULT_DATA_PATH
from .seq_modeling.downloads import download_pretrained_model
from .seq_modeling.helpers import load_vocab_dict, get_model_nparams
from .util import is_module_available


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


class Corrector(ABC):
    DEFAULT_CHECKPOINT_PATH = {
        "bertscrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/bertscrnn-probwordnoise",
        "cnn-lstm-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/cnn-lstm-probwordnoise",
        "lstm-lstm-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/lstm-lstm-probwordnoise",
        "scrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnn-probwordnoise",
        "scrnnbert-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnnbert-probwordnoise",
        "subwordbert-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/subwordbert-probwordnoise",
    }
    if is_module_available("allennlp"):
        DEFAULT_CHECKPOINT_PATH.update({
            "elmoscrnn-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/elmoscrnn-probwordnoise",
            "scrnnelmo-probwordnoise": f"{DEFAULT_DATA_PATH}/checkpoints/scrnnelmo-probwordnoise",
        })

    # TODO: deprecated usage; should be reoved in next versions
    DEFAULT_CHECKERNAME_TO_NAME_MAPPING = {
        "BertsclstmChecker": "bertscrnn-probwordnoise",
        "CnnlstmChecker": "cnn-lstm-probwordnoise",
        "NestedlstmChecker": "lstm-lstm-probwordnoise",
        "SclstmChecker": "scrnn-probwordnoise",
        "SclstmbertChecker": "scrnnbert-probwordnoise",
        "BertChecker": "subwordbert-probwordnoise",
    }
    if is_module_available("allennlp"):
        DEFAULT_CHECKERNAME_TO_NAME_MAPPING.update({
            "ElmosclstmChecker": "elmoscrnn-probwordnoise",
            "SclstmelmoChecker": "scrnnelmo-probwordnoise",
        })

    def __init__(self, **kwargs):

        self._default_name = kwargs.get("name", None)
        self.tokenize = kwargs.get("tokenize", True)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if self.device == "gpu" else self.device

        self.ckpt_path, self.vocab_path, self.weights_path = None, None, None
        self.model, self.vocab = None, None

        if not self._default_name:

            try:
                self._default_name = Corrector.DEFAULT_CHECKERNAME_TO_NAME_MAPPING[self.__class__.__name__]
            except KeyError as e:
                msg = f"Unable to resolve checker name {self.__class__.__name__} " \
                      f"from list of known names {[*NAME_TO_CHECKER_MAPPINGS.keys()]}"
                raise ModuleNotFoundError(msg) from e

        if kwargs.get("pretrained", False):
            self.from_pretrained(ckpt_path=self.ckpt_path)

    def is_model_ready(self):
        assert not (self.model is None or self.vocab is None), print("model & vocab must be loaded first")

    @property
    def get_device(self):
        return self.device

    def set_device(self, device='cpu'):
        prev_device = self.device
        device = "cuda" if ((device == "gpu" or device == "cuda") and torch.cuda.is_available()) else "cpu"
        if not (prev_device == device):
            # use .to() if moving from cpu or gpu, and for reverse, use map_location
            # https://tinyurl.com/y57pcjvd
            # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
            if self.model is not None:
                try:
                    self.model.to(device)
                except Exception as e:
                    try:
                        self.from_pretrained(self.ckpt_path, vocab=self.vocab_path)
                    except Exception as e:
                        msg = f"Unable to move model from {prev_device} to {device}. " \
                              f"Please load a new instance with argument `device={device}. "
                        raise Exception(msg)
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

    def correct_from_file(self, src, dest="./clean_version.txt"):
        self.is_model_ready()
        x = [line.strip() for line in open(src, 'r')]
        y = self.correct_strings(x)
        print(f"saving results at: {dest}")
        opfile = open(dest, 'w')
        for line in y:
            opfile.write(line + "\n")
        opfile.close()
        return

    def _from_pretrained(self, ckpt_path=None, vocab_path=None):

        if ckpt_path:
            self._default_name = os.path.split(ckpt_path)[-1]
            self.ckpt_path = ckpt_path
        else:
            # self._default_name is kept default
            self.ckpt_path = Corrector.DEFAULT_CHECKPOINT_PATH[self._default_name]

        self.vocab_path = vocab_path or os.path.join(self.ckpt_path, "vocab.pkl")
        if not os.path.isfile(self.vocab_path):  # leads to "FileNotFoundError"
            download_pretrained_model(self.ckpt_path)

        self.load_output_vocab(self.vocab_path)
        self.load_model(self.ckpt_path)

        return

    def from_pretrained(self, ckpt_path=None, vocab_path=None, **kwargs):
        print("calling from_pretrained")
        self._from_pretrained(ckpt_path, vocab_path, **kwargs)

    def load_output_vocab(self, vocab_path):
        print(f"loading vocab from path:{vocab_path}")
        self.vocab = load_vocab_dict(vocab_path)

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def finetune(self, *args, **kwargs):
        raise NotImplementedError

    def add_(self, contextual_model, at="input"):
        raise Exception("this functionality is only available with `SclstmChecker`")

    @abstractmethod
    def load_model(self, ckpt_path):
        raise NotImplementedError

    @abstractmethod
    def correct_strings(self, mystrings: List[str], return_all=False):
        raise NotImplementedError

    @property
    def get_num_params(self):
        self.is_model_ready()
        return get_model_nparams(self.model)

    def model_size(self, model=None):
        if not model:
            model = self.model
            self.is_model_ready()
        sz = {
            "num_params": get_model_nparams(model),
            "disk_size (in MB)": get_size_of_model(model),
        }
        return sz

    # new!!
    def quantize_model(self, print_stats=False):
        self.is_model_ready()

        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except RuntimeError as e:
            msg = "Consider moving models to `cpu` by calling `.set_device(device='cpu')` before quantization. "
            raise Exception(msg) from e

        if print_stats:
            print("Before quantization:")
            print(self.model_size())
            print("After quantization:")
            print(self.model_size(quantized_model))

        self.model = quantized_model
