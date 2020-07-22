"""Class wrapping Hunspell checker.

Installation:
    pip install CyHunspell==1.2.1

Usage:
    ```
    checker = HunspellChecker()
    corrected_word = checker.correct_word(word)
    corrected_sentence = checker.correct_string(sentence)
    ```
"""

import string
from nltk.corpus import stopwords as SW
from hunspell import Hunspell
from utils_ import read_birkbeck
from tqdm import tqdm

class HunspellChecker(object):
    def __init__(self):
        self.checker = Hunspell()
        self.stopwords = set(SW.words("english")) | set(string.punctuation)

    def correct_word(self, word):
        """Borrowed from:
        https://datascience.blog.wzb.eu/2016/07/13/autocorrecting-misspelled-words-in-python-using-hunspell/
        """
        ok = self.checker.spell(word)   # check spelling
        if not ok:
            suggestions = self.checker.suggest(word)
            if len(suggestions) > 0:  # there are suggestions
                return suggestions[0]
            else:
                return word
        else:
            return word

    def correct_string(self, text, ensure_length=False):
        """Break into words and correct each word."""
        tokens = text.split()
        corrected = []
        for token in tokens:
            if token in self.stopwords: corrected.append(token)
            else:
                correction = self.correct_word(token)
                if ensure_length:
                    corrected.append(correction.split()[0])
                else:
                    corrected.append(correction)
        return " ".join(corrected)

if __name__ == "__main__":
    checker = HunspellChecker()

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
