import string
from nltk.corpus import stopwords as SW
from collections import defaultdict
import random
from random import shuffle
import numpy as np

punctuations = string.punctuation +  ' '
stopwords = set(SW.words("english")) | set(string.punctuation)
keyboard_mappings = None

MIN_LEN = 5

def drop_one_attack_deprecated(line):
    for i in range(1, len(line) - 1):
        if line[i] in punctuations or line[i-1] in punctuations \
        or line[i+1] in punctuations:
            # first or last character of a word
            continue

        # drop the ith character
        new_line = line[:i] + line[i+1:]
        if len(new_line.split()) != len(line.split()):
            # probably dropped a single char word
            continue
        yield new_line


def drop_one_attack(line, ignore_indices = set(), include_ends=False):
    """ an attack that drops one character at a time

    Arguments:
        line {string} -- the input line/review/comment

    Keyword Arguments:
        ignore_idx {int} -- ignores a given set of indices
        include_ends {bool} -- to include dropping the first & last char?
    """

    words = line.split()

    for idx, word in enumerate(words):

        if len(word) < 3: continue
        if word in stopwords: continue
        if idx in ignore_indices: continue

        if include_ends:
            adversary_words = [word[:i] + word[i+1:] for i in range(0, len(word))]
        else:
            adversary_words = [word[:i] + word[i+1:] for i in range(1, len(word)-1)]

        for adv in adversary_words:
            yield idx, " ".join(words[:idx] + [adv] + words[idx+1:])


def swap_one_attack(line, ignore_indices = set(), include_ends=False):
    """ an attack that drops one character at a time

    Arguments:
        line {string} -- the input line/review/comment

    Keyword Arguments:
        ignore_idx {int} -- ignores a given set of indices
        include_ends {bool} -- to include dropping the first & last char?
    """

    words = line.split()

    for idx, word in enumerate(words):

        if len(word) < MIN_LEN: continue
        if word in stopwords: continue
        if idx in ignore_indices: continue

        if include_ends:
            adversary_words = [word[:i] + word[i:i+2][::-1] + word[i+2:] for i in range(0, len(word)-1)]
        else:
            adversary_words = [word[:i] + word[i:i+2][::-1] + word[i+2:] for i in range(1, len(word)-2)]

        for adv in adversary_words:
            yield idx, " ".join(words[:idx] + [adv] + words[idx+1:])



def add_one_attack(line, ignore_indices = set(), include_ends=False,
                   alphabets="abcdefghijklmnopqrstuvwxyz"):
    """ an attack that adds one random character in the line

    Arguments:
        line {string} -- the input line/review/comment

    Keyword Arguments:
        ignore_idx {int} -- ignores a given set of indices
        include_ends {bool} -- include the first & last char for adddition?
    """

    words = line.split()

    #alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]

    for idx, word in enumerate(words):

        if len(word) < 3: continue # need to have this same as defense settings
        if word in stopwords: continue
        if idx in ignore_indices: continue

        if include_ends:
            adversary_words = [word[:i] + alpha + word[i:] for i in range(0, len(word) + 1) \
                                for alpha in alphabets]
        else:
            adversary_words = [word[:i] + alpha + word[i:] for i in range(1, len(word)) \
                                for alpha in alphabets]

        for adv in adversary_words:
            yield idx, " ".join(words[:idx] + [adv] + words[idx+1:])


def key_one_attack(line, ignore_indices = set(), include_ends=False):
    """ an attack that adds one random character in the line

    Arguments:
        line {string} -- the input line/review/comment

    Keyword Arguments:
        ignore_idx {int} -- ignores a given set of indices
        include_ends {bool} -- include the first & last char for adddition?
    """

    words = line.split()

    for idx, word in enumerate(words):

        if len(word) < 3: continue # need to have this same as defense settings
        if word in stopwords: continue
        if idx in ignore_indices: continue

        adversary_words = []
        if include_ends:
            for i in range(0, len(word)):
                for key in get_keyboard_neighbors(word[i]):
                    adversary_words.append(word[:i] + key + word[i+1:])
        else:
            for i in range(1, len(word) - 1):
                for key in get_keyboard_neighbors(word[i]):
                    adversary_words.append(word[:i] + key + word[i+1:])

        for adv in adversary_words:
            yield idx, " ".join(words[:idx] + [adv] + words[idx+1:])


# this is the all attack setting, where all of the add/swap/drop/key
# attacks are tried
def all_one_attack(line, ignore_indices = set(), include_ends=False,
                   alphabets="abcdefghijklmnopqrstuvwxyz"):
	generator = add_one_attack(line, ignore_indices, include_ends,
                            alphabets=alphabets)
	for idx, adv in generator:
		yield idx, adv
	generator = key_one_attack(line, ignore_indices, include_ends)
	for idx, adv in generator:
		yield idx, adv
	generator = drop_one_attack(line, ignore_indices, include_ends)
	for idx, adv in generator:
		yield idx, adv
	generator = swap_one_attack(line, ignore_indices, include_ends)
	for idx, adv in generator:
		yield idx, adv


def random_all_one_attack(line, ignore_indices=set(), include_ends=False):
    generators = [add_one_attack, key_one_attack, drop_one_attack, swap_one_attack]
    shuffle(generators)
    for generator in generators:
        for idx, adv in generator(line, ignore_indices, include_ends):
            yield idx, adv


def get_keyboard_neighbors(ch):
    global keyboard_mappings
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return [ch]
    return keyboard_mappings[ch]


def is_valid_attack(line, char_idx):
    line = line.lower()
    if char_idx == 0 or char_idx == len(line) - 1:
        # first and last chars of the sentence
        return False
    if line[char_idx-1] == ' ' or line[char_idx+1] == ' ':
        # first and last chars of the word
        return False
    # anything not a legit alphabet
    if not('a' <= line[char_idx] <= 'z'):
        return False
    return True


def get_random_attack(line):
    num_chars = len(line)
    NUM_TRIES = 10

    for _ in range(NUM_TRIES):
        char_idx = np.random.choice(range(num_chars), 1)[0]
        if is_valid_attack(line, char_idx):
            attack_type = ['swap', 'drop', 'add', 'key']
            attack_probs = np.array([1.0, 1.0, 10.0, 2.0])
            attack_probs = attack_probs/sum(attack_probs)
            attack = np.random.choice(attack_type, 1, p=attack_probs)[0]
            if attack == 'swap':
                return line[:char_idx] + line[char_idx:char_idx+2][::-1] + line[char_idx+2:]
            elif attack == 'drop':
                return line[:char_idx] + line[char_idx+1:]
            elif attack == 'key':
                sideys = get_keyboard_neighbors(line[char_idx])
                new_ch = np.random.choice(sideys, 1)[0]
                return line[:char_idx] + new_ch + line[char_idx+1:]
            else: # attack type is add
                alphabets = "abcdefghijklmnopqrstuvwxyz"
                alphabets = [ch for ch in alphabets]
                new_ch = np.random.choice(alphabets, 1)[0]
                return line[:char_idx] + new_ch + line[char_idx:]
    return None