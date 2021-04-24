from typing import List

from .commons import spacy_tokenizer
from .corrector import Corrector
from .seq_modeling.helpers import load_data
from .seq_modeling.util import is_module_available

if is_module_available("allennlp"):
    from .seq_modeling.sclstmelmo import load_model, load_pretrained, model_predictions, model_inference

""" corrector module """


class SclstmelmoChecker(Corrector):

    def __init__(self, **kwargs):

        if not is_module_available("allennlp"):
            raise ImportError(
                "install `allennlp` by running `pip install -r extras-requirements.txt`. "
                "See `README.md` for more info.")

        super().__init__(**kwargs)

    def load_model(self, ckpt_path):
        print(f"initializing model")
        initialized_model = load_model(self.vocab)
        self.model = load_pretrained(initialized_model, self.ckpt_path, device=self.device)

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.is_model_ready()
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 16
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def evaluate(self, clean_file, corrupt_file, data_dir=""):
        self.is_model_ready()
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir

        batch_size = 4 if self.device == "cpu" else 16
        for x, y, z in zip([data_dir], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            _ = model_inference(self.model,
                                test_data,
                                topk=1,
                                device=self.device,
                                batch_size=batch_size,
                                vocab_=self.vocab)
        return
