"""
# some methods of this file are obtained from Danish et al. 2019
# it does 4 types of edits for words with len>2
#   swap, add, drop, key
#   only one of these operations is applied to any word at a time
#   the selection of one of these opertaions for any word is random, with equal chances
"""

import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

keyboard_mappings = None


def _get_random_char():
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    return np.random.choice(alphabets, 1)[0]


def _get_keyboard_neighbor(ch):
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

    if ch not in keyboard_mappings: return ch
    return np.random.choice(keyboard_mappings[ch], 1)[0]


def _get_swap_word_representation(word):
    # dirty case
    if len(word) == 1 or len(word) == 2:
        # rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1])
        # return rep, word
        return word

    # rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1])
    if len(word) > 3:
        idx = random.randint(1, len(word) - 3)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]

    # return rep, word
    return word


def _get_drop_word_representation(word, prob=0.5):
    p = random.random()
    if len(word) >= 5 and p < prob:
        idx = random.randint(1, len(word) - 2)
        word = word[:idx] + word[idx + 1:]
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    elif p > prob:
        # rep, word = _get_swap_word_representation(word)
        word = _get_swap_word_representation(word)
    else:
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    # return rep, word
    return word


def _get_add_word_representation(word):
    if len(word) >= 3:
        idx = random.randint(1, len(word) - 1)
        random_char = _get_random_char()
        word = word[:idx] + random_char + word[idx:]
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    else:
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    # return rep, word
    return word


def _get_keyboard_word_representation(word):
    if len(word) >= 3:
        idx = random.randint(1, len(word) - 2)
        keyboard_neighbor = _get_keyboard_neighbor(word[idx])
        word = word[:idx] + keyboard_neighbor + word[idx + 1:]
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    else:
        # rep, _ = _get_swap_word_representation(word) # don't care about the returned word
        _ = _get_swap_word_representation(word)  # don't care about the returned word
    # return rep, word
    return word


def _get_line_representation(line, rep_list, probs):
    modified_words = []
    for word in line.split():
        rep_type = np.random.choice(rep_list, 1, p=probs)[0]
        if 'swap' in rep_type:
            new_word = _get_swap_word_representation(word)
        elif 'drop' in rep_type:
            new_word = _get_drop_word_representation(word, 1.0)
        elif 'add' in rep_type:
            new_word = _get_add_word_representation(word)
        elif 'key' in rep_type:
            new_word = _get_keyboard_word_representation(word)
        elif 'none' in rep_type or 'normal' in rep_type:
            new_word = word
        else:
            raise NotImplementedError
        modified_words.append(new_word)
    return " ".join(modified_words)


def get_line_representation(lines,
                            rep_list=['swap', 'drop', 'add', 'key', 'none'],
                            probs=[0.1, 0.1, 0.1, 0.1, 0.6],
                            verbose=False):
    if verbose:
        print(f"the kinds of replacements and their probabilities are: {tuple(zip(rep_list, probs))}")

    modified_lines = []
    for line in tqdm(lines):
        modified_lines.append(_get_line_representation(line, rep_list, probs))

    return modified_lines
