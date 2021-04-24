import sys
from time import time
from typing import List

import jamspell
from tqdm import tqdm

sys.path.append("/..")
from scripts.seq_modeling.helpers import load_data
from commons import spacy_tokenizer
from scripts.evals import get_metrics

""" corrector module """


class JamspellChecker:

    def __init__(self, tokenize=True):
        self.tokenize = tokenize
        self.model = jamspell.TSpellCorrector()
        self.model.LoadLangModel('en.bin')

    def from_pretrained(self, ckpt_path, vocab="", weights=""):
        return

    def set_device(self, device='cpu'):
        print(f"model set to work on cpu")
        return

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        return_strings = []
        for line in tqdm(mystrings):
            new_line = self.model.FixFragment(line)
            return_strings.append(new_line)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def evaluate(self, clean_file, corrupt_file):
        """
        clean_file = f"{DEFAULT_DATA_PATH}/traintest/clean.txt"
        corrupt_file = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        for x, y, z in zip([""], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            clean_data = [x[0] for x in test_data]
            corrupt_data = [x[1] for x in test_data]
            inference_st_time = time()
            predictions_data = self.correct_strings(corrupt_data)
            assert len(clean_data) == len(corrupt_data) == len(predictions_data)
            corr2corr, corr2incorr, incorr2corr, incorr2incorr = \
                get_metrics(clean_data, corrupt_data, predictions_data)

            print("total inference time for this data is: {:4f} secs".format(time() - inference_st_time))
            print("###############################################")
            print("total token count: {}".format(corr2corr + corr2incorr + incorr2corr + incorr2incorr))
            print(
                f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
            print(f"accuracy is {(corr2corr + incorr2corr) / (corr2corr + corr2incorr + incorr2corr + incorr2incorr)}")
            print(f"word correction rate is {(incorr2corr) / (incorr2corr + incorr2incorr)}")
            print("###############################################")

        return

    def model_size(self):
        return None
