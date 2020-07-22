"""Class wrapping AfterTheDeadline checker.

Installation:
Make sure you have the ATD package downloaded and running as
a server on localhost.

Download the source and models from:
http://www.polishmywriting.com/download/atd_distribution081310.tgz

Then follow the installation / test instructions at:
https://open.afterthedeadline.com/how-to/get-started/

This will run the ATD server on localhost at 127.0.0.1:1049.

Usage:
    ```
    checker = ATDChecker()
    corrected_word = checker.correct_word(word)
    corrected_sentence = checker.correct_string(sentence)
    ```
"""

import ATD
from tqdm import tqdm

birkbeck_data = "../../data/missp.dat"

def read_birkbeck():
    """Returns pairs of (incorrect, correct) spellings."""
    misspellings = []
    with open(birkbeck_data) as f:
        correct = None
        incorrect = []
        for line in f:
            word = line.strip()
            if word.startswith("$"):
                # new group
                if correct is not None:
                    misspellings += [(correct, ii) for ii in incorrect]
                    incorrect = []
                correct = word[1:]
            else:
                incorrect += [word]
    return misspellings


class ATDChecker(object):
    def __init__(self):
        ATD.setDefaultKey("break-it")

    def correct_word(self, word):
        errors = ATD.checkDocument(word)
        for error in errors:
            if error.description == "Spelling" and error.suggestions:
                return error.suggestions[0]
        return word

    def correct_string(self, text, ensure_length=False):
        tokens = text.split()
        errors = ATD.checkDocument(text)
        subs = {}
        for error in list(errors):
            l_suggestions = list(error.suggestions)
            if error.description == "Spelling" and len(l_suggestions) > 0:
                subs[error.string] = l_suggestions[0]
        for i, t in enumerate(list(tokens)):
            if t in subs:
                if ensure_length:
                    tokens[i] = subs[t].split()[0]
                else:
                    tokens[i] = subs[t]
        return " ".join(tokens)

def test_basic():
    checker = ATDChecker()
    test_data = [
        "The bal has gone out of play",
        "He is standing near the waer",
        "Wow whatt a gamme",
        "Mo Salah is a genius"
    ]

    for item in test_data:
        print("Original: ", item)
        print("Corrected: ", checker.correct_string(item))

def test_birkbeck():
    checker = ATDChecker()
    test_data = read_birkbeck()
    fixed, total = 0., 0
    mistakes = []
    for correct, incorrect in tqdm(test_data):
        correction = checker.correct_word(incorrect)
        if correction.lower() != correct.lower():
            mistakes.append((correct, incorrect, correction))
        else:
            fixed += 1.
        total += 1
    print("Accuracy = %.3f" % (fixed / total))
    print("Mistakes:")
    print("Correct\tIncorrect\tCorrection")
    print("\n".join("%s\t%s\t%s" % (a,b,c) for a,b,c in mistakes[:10]))

if __name__ == "__main__":
    test_birkbeck()
