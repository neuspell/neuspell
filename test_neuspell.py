"""
USAGE
-----
for gpu (+- cpu) testing:
>>> CUDA_VISIBLE_DEVICES=0 python test_neuspell.py
for cpu-only testing:
>>> python test_neuspell.py
-----
"""

# from neuspell import AspellChecker, JamspellChecker
from neuspell import CnnlstmChecker, BertsclstmChecker, ElmosclstmChecker, NestedlstmChecker
from neuspell import SclstmbertChecker, SclstmelmoChecker, BertChecker, SclstmChecker

traintest_path = "./data/traintest"
Checker = SclstmChecker

######################################################
######################################################


""" load a checker from a checkpoint; defaults to load on cpu device """
checker = Checker(pretrained=True)
print("to cheque sum spelling rul", "\n\t\t→", checker.correct("to cheque sum spelling rul"))
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt", dest=f"{traintest_path}/sample_prediction.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")

""" move loaded checker to work on gpu """
checker.set_device('gpu')
print("I lok forward to receving ur reply", "\n\t\t→", checker.correct("I lok forward to receving ur reply"))
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt", dest=f"{traintest_path}/sample_prediction.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")

""" move back the checker to work on cpu """
checker.set_device('cpu')
print("misteaks eye can knot sea", "\n\t\t→", checker.correct("misteaks eye can knot sea"))
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt", dest=f"{traintest_path}/sample_prediction.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")

######################################################
######################################################

""" select a checker and load it from a checkpoint; defaults to load on cpu device """
if Checker == SclstmChecker:
    checker.from_pretrained("./data/checkpoints/scrnn-probwordnoise")
    print("model size: ", checker.model_size())
    print("It shows me strait a weigh as soon as a mistache is maid .", "\n\t\t→",
          checker.correct("It shows me strait a weigh as soon as a mistace is maid ."))

######################################################
######################################################

""" add elmo/bert at input/output; currently available only for SclstmChecker """
if Checker == SclstmChecker:
    checker = Checker()
    checker = checker.add_("elmo", at="input")
    # checker.from_pretrained("./data/checkpoints/elmoscrnn-probwordnoise")
    print("model size: ", checker.model_size())
    print("It shows me strait a weigh as soon as a mistache is maid .", "\n\t\t→",
          checker.correct("It shows me strait a weigh as soon as a mistace is maid ."))

######################################################
######################################################
