"""
USAGE
-----
for gpu (+- cpu) testing:
>>> CUDA_VISIBLE_DEVICES=0 python test.py
for cpu-only testing:
>>> python test.py
-----
"""

from neuspell.util import is_module_available
import logging

# from neuspell import AspellChecker, JamspellChecker
from neuspell import CnnlstmChecker, BertsclstmChecker, NestedlstmChecker, SclstmbertChecker, BertChecker, SclstmChecker

all_checkers = [
    CnnlstmChecker,
    BertsclstmChecker,
    NestedlstmChecker,
    SclstmbertChecker,
    BertChecker,
    SclstmChecker
]

if is_module_available("allennlp"):
    from neuspell import ElmosclstmChecker, SclstmelmoChecker

    all_checkers.append([ElmosclstmChecker, SclstmelmoChecker])

logging.getLogger().setLevel(logging.ERROR)
TRAIN_TEST_DATA_PATH = "./data/traintest"

######################################################
######################################################

# for Checker in all_checkers:
#     print("\n######################################################")
#     print(f"checking {Checker.__name__}")
#
#     """ load a checker from a checkpoint; defaults to load on cpu device """
#     checker = Checker()
#     checker.from_pretrained()
#     print("to cheque sum spelling rul", "\n\t\t→", checker.correct("to cheque sum spelling rul"))
#     checker.correct_from_file(src=f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt",
#                               dest=f"{TRAIN_TEST_DATA_PATH}/sample_prediction.txt")
#     checker.evaluate(f"{TRAIN_TEST_DATA_PATH}/sample_clean.txt", f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt")
#
#     """ load a checker from a checkpoint; defaults to load on cpu device """
#     checker = Checker(pretrained=True)
#     print("to cheque sum spelling rul", "\n\t\t→", checker.correct("to cheque sum spelling rul"))
#     checker.correct_from_file(src=f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt",
#                               dest=f"{TRAIN_TEST_DATA_PATH}/sample_prediction.txt")
#     checker.evaluate(f"{TRAIN_TEST_DATA_PATH}/sample_clean.txt", f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt")
#
#     """ move loaded checker to work on gpu """
#     checker.set_device('gpu')
#     print("I lok forward to receving ur reply", "\n\t\t→", checker.correct("I lok forward to receving ur reply"))
#     checker.correct_from_file(src=f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt",
#                               dest=f"{TRAIN_TEST_DATA_PATH}/sample_prediction.txt")
#     checker.evaluate(f"{TRAIN_TEST_DATA_PATH}/sample_clean.txt", f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt")
#
#     """ move back the checker to work on cpu """
#     checker.set_device('cpu')
#     print("misteaks eye can knot sea", "\n\t\t→", checker.correct("misteaks eye can knot sea"))
#     checker.correct_from_file(src=f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt",
#                               dest=f"{TRAIN_TEST_DATA_PATH}/sample_prediction.txt")
#     checker.evaluate(f"{TRAIN_TEST_DATA_PATH}/sample_clean.txt", f"{TRAIN_TEST_DATA_PATH}/sample_corrupt.txt")
#
#     print("######################################################\n")

######################################################
######################################################


if SclstmChecker in all_checkers:
    """ select a checker and load it from a checkpoint; defaults to load on cpu device """
    checker = SclstmChecker()
    checker.from_pretrained("./data/checkpoints/scrnn-probwordnoise")
    print("model size: ", checker.model_size())
    print("It shows me strait a weigh as soon as a mistache is maid .", "\n\t\t→",
          checker.correct("It shows me strait a weigh as soon as a mistace is maid ."))

    """ add elmo/bert at input/output; currently available only for SclstmChecker """
    checker = checker.add_("elmo", at="input")  # loads model as well!
    print("model size: ", checker.model_size())
    print("It shows me strait a weigh as soon as a mistache is maid .", "\n\t\t→",
          checker.correct("It shows me strait a weigh as soon as a mistace is maid ."))

######################################################
######################################################
