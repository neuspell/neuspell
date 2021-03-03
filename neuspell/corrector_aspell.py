"""
USAGE
---
python model_aspell.py ../../data/traintest/test.bea4k ../../data/traintest/test.bea4k.noise
python model_aspell.py ../../data/traintest/test.1blm ../../data/traintest/test.1blm.noise.word
python model_aspell.py ../../data/traintest/test.1blm ../../data/traintest/test.1blm.noise.prob

python model_aspell.py ../../data/traintest/wo_context/aspell_small ../../data/traintest/wo_context/aspell_small.noise
python model_aspell.py ../../data/traintest/wo_context/aspell_big ../../data/traintest/wo_context/aspell_big.noise
python model_aspell.py ../../data/traintest/wo_context/combined_data ../../data/traintest/wo_context/combined_data.noise
python model_aspell.py ../../data/traintest/wo_context/homophones ../../data/traintest/wo_context/homophones.noise

USAGE
---
from corrector_aspell import CorrectorAspell
correctorAspell = CorrectorAspell()
predictions_data = correctorAspell.correct_string("nicset atcing I have ever witsesed")
"""

from time import time
from typing import List

import aspell
from tqdm import tqdm

from .commons import spacy_tokenizer, Corrector
from .seq_modeling.evals import get_metrics
from .seq_modeling.helpers import load_data

""" corrector module """


class CorrectorAspell(Corrector):
    def __init__(self, tokenize=True, pretrained=False, device="cpu"):
        super(CorrectorAspell, self).__init__()
        self.tokenize = tokenize
        self.pretrained = None
        self.device = None

        self.ckpt_path = None
        self.vocab_path, self.weights_path = "", ""
        self.model, self.vocab = None, None

        self.model = aspell.Speller()
        self.model.setConfigKey('sug-mode', "normal")  # ultra, fast, normal, slow, or bad-spellers

    def from_pretrained(self, ckpt_path, vocab="", weights=""):
        return

    def set_device(self, device='cpu'):
        print(f"model set to work on cpu")
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
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        return_strings = []
        for line in tqdm(mystrings):
            new_line = []
            for corrupt_token in line.strip().split():
                suggestions = self.model.suggest(corrupt_token)
                if len(suggestions) > 0 and suggestions[0] != "":
                    new_line.append(suggestions[0])
                else:
                    # new_line.append("<<EMPTY>>")
                    new_line.append(corrupt_token)
            new_line = " ".join(new_line)
            return_strings.append(new_line)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def correct_from_file(self, src, dest="./clean_version.txt"):
        """
        src = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
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

# if __name__ == "__main__":
#
#     TEMP_FOLDER = "./aspell_temp"
#     if not os.path.exists(TEMP_FOLDER): os.makedirs(TEMP_FOLDER)
#
#     CLEAN_FILE_PATH = sys.argv[1]
#     CORRUPT_FILE_PATH = sys.argv[2]
#     PREDICTION_FILE_PATH = os.path.join(TEMP_FOLDER, CORRUPT_FILE_PATH.split("/")[-1] + ".prediction")
#
#     opfile = open(CLEAN_FILE_PATH, "r")
#     clean_data = opfile.readlines()
#     opfile.close()
#     print("total lines in clean_data: {}".format(len(clean_data)))
#     print("total tokens in clean_data: {}".format(sum([len(line.strip().split()) for line in clean_data])))
#
#     opfile = open(CORRUPT_FILE_PATH, "r")
#     corrupt_data = opfile.readlines()
#     opfile.close()
#     print("total lines in corrupt_data: {}".format(len(corrupt_data)))
#     print("total tokens in corrupt_data: {}".format(sum([len(line.strip().split()) for line in corrupt_data])))
#
#     assert len(clean_data) == len(corrupt_data)
#
#     correctorAspell = CorrectorAspell()
#     predictions_data = correctorAspell.correct_strings(corrupt_data)
#
#     assert len(clean_data) == len(corrupt_data) == len(predictions_data)
#
#     corr2corr, corr2incorr, incorr2corr, incorr2incorr = \
#         get_metrics(clean_data, corrupt_data, predictions_data)
#
#     opfile = open(PREDICTION_FILE_PATH, "w")
#     for line in predictions_data[:-1]:
#         opfile.write("{}\n".format(line))
#     opfile.write("{}".format(predictions_data[-1]))
#     opfile.close()
