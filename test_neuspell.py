
"""
USAGE
-----
for gpu (+- cpu) testing:
	CUDA_VISIBLE_DEVICES=0 python test_neuspell.py
for cpu testing only:
	python test_neuspell.py
-----
"""
from neuspell import AspellChecker, JamspellChecker
from neuspell import CnnlstmChecker, SclstmChecker, NestedlstmChecker, BertChecker
from neuspell import ElmosclstmChecker, SclstmelmoChecker
from neuspell import BertsclstmChecker, SclstmbertChecker

traintest_path = "./data/traintest"

TestCheker = ElmosclstmChecker

checker = TestCheker(pretrained=False)
checker = TestCheker(pretrained=True)

print(checker.correct("I lok forward to receving yur reply"))
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt",  dest=f"{traintest_path}/sample_predicton.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")

checker.set_device('gpu')
print(checker.correct("I lok forward to receving ur reply"))
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt",  dest=f"{traintest_path}/sample_predicton.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")

checker.set_device('cpu')
checker.correct_from_file(src=f"{traintest_path}/sample_corrupt.txt",  dest=f"{traintest_path}/sample_predicton.txt")
checker.evaluate(f"{traintest_path}/sample_clean.txt", f"{traintest_path}/sample_corrupt.txt")