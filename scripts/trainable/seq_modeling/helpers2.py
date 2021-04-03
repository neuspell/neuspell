""" helper functions for
    - data loading
    - representation building
    - vocabulary loading
"""

from collections import defaultdict
import numpy as np
# import pickle
import random
# from random import shuffle
from tqdm import tqdm

############################################################

# #TODO: think of an open vocabulary system
# WORD_LIMIT = 9999 # remaining 1 for <PAD> (this is inclusive of UNK)
# task_name = ""
# TARGET_PAD_IDX = -1
# INPUT_PAD_IDX = 0

keyboard_mappings = None

############################################################

def get_lines(filename):
    print(filename)
    f = open(filename)
    lines = f.readlines()
    if "|||" in lines[0]:
        # remove the tag
        clean_lines = [line.split("|||")[1].strip().lower() for line in lines]
    else:
        clean_lines = [line.strip().lower() for line in lines]
    return clean_lines

############################################################

# CHAR_VOCAB = []
# w2i = defaultdict(lambda: 0.0)
# i2w = defaultdict(lambda: "UNK")

# def create_vocab(filename, background_train=False, cv_path=""):
#     global w2i, i2w, CHAR_VOCAB
#     lines = get_lines(filename)
#     for line in lines:
#         for word in line.split():

#             # add all its char in vocab
#             for char in word:
#                 if char not in CHAR_VOCAB:
#                     CHAR_VOCAB.append(char)

#             w2i[word] += 1.0

#     if background_train:
#         CHAR_VOCAB = pickle.load(open(cv_path, 'rb'))
#     word_list = sorted(w2i.items(), key=lambda x:x[1], reverse=True)
#     word_list = word_list[:WORD_LIMIT] # only need top few words

#     # remaining words are UNKs ... sorry!
#     w2i = defaultdict(lambda: WORD_LIMIT) # default id is UNK ID
#     w2i['<PAD>'] = INPUT_PAD_IDX # INPUT_PAD_IDX is 0
#     i2w[INPUT_PAD_IDX] = '<PAD>'
#     for idx in range(WORD_LIMIT-1):
#         w2i[word_list[idx][0]] = idx+1
#         i2w[idx+1] = word_list[idx][0]

#     pickle.dump(dict(w2i), open("vocab/" + task_name + "w2i_" + str(WORD_LIMIT) + ".p", 'wb'))
#     pickle.dump(dict(i2w), open("vocab/" + task_name + "i2w_" + str(WORD_LIMIT) + ".p", 'wb')) # don't think its needed
#     pickle.dump(CHAR_VOCAB, open("vocab/" + task_name + "CHAR_VOCAB_ " + str(WORD_LIMIT) + ".p", 'wb'))
#     return

############################################################

# CHAR_VOCAB_BG = []
# w2i_bg = defaultdict(lambda: 0.0)
# i2w_bg = defaultdict(lambda: "UNK")

# def load_vocab_dicts(wi_path, iw_path, cv_path, use_background=False):
#     wi = pickle.load(open(wi_path, 'rb'))
#     iw = pickle.load(open(iw_path, 'rb'))
#     cv = pickle.load(open(cv_path, 'rb'))
#     if use_background:
#         convert_vocab_dicts_bg(wi, iw, cv)
#     else:
#         convert_vocab_dicts(wi, iw, cv)

# """ 
# converts vocabulary dictionaries into defaultdicts
# """
# def convert_vocab_dicts(wi, iw, cv):
#     global w2i, i2w, CHAR_VOCAB
#     CHAR_VOCAB = cv
#     w2i = defaultdict(lambda: WORD_LIMIT)
#     for w in wi:
#         w2i[w] = wi[w]

#     for i in iw:
#         i2w[i] = iw[i]
#     return

# def convert_vocab_dicts_bg(wi, iw, cv):
#     global w2i_bg, i2w_bg, CHAR_VOCAB_BG
#     CHAR_VOCAB_BG = cv
#     w2i_bg = defaultdict(lambda: WORD_LIMIT)
#     for w in wi:
#         w2i_bg[w] = wi[w]

#     for i in iw:
#         i2w_bg[i] = iw[i]
#     return

############################################################

# def get_target_representation(line):
#     return [w2i[word] for word in line.split()]

# def pad_input_sequence(X, max_len):
#     assert (len(X) <= max_len)
#     while len(X) != max_len:
#         X.append([INPUT_PAD_IDX for _ in range(len(X[0]))])
#     return X

# def pad_target_sequence(y, max_len):
#     assert (len(y) <= max_len)
#     while len(y) != max_len:
#         y.append(TARGET_PAD_IDX)
#     return y

# def get_batched_input_data(lines, batch_size, rep_list=['swap'], probs=[1.0]):
#     #shuffle(lines)
#     total_len = len(lines)
#     output = []
#     for batch_start in range(0, len(lines) - batch_size, batch_size):

#         input_lines = []
#         modified_lines = []
#         X = []
#         y = []
#         lens = []
#         max_len = max([len(line.split()) \
#                 for line in lines[batch_start: batch_start + batch_size]])

#         for line in lines[batch_start: batch_start + batch_size]:
#             X_i, modified_line_i = get_line_representation(line, rep_list, probs)
#             assert (len(line.split()) == len(modified_line_i.split()))
#             y_i = get_target_representation(line)
#             # pad X_i, and y_i
#             X_i = pad_input_sequence(X_i, max_len)
#             y_i = pad_target_sequence(y_i, max_len)
#             # append input lines, modified lines, X_i, y_i, lens
#             input_lines.append(line)
#             modified_lines.append(modified_line_i)
#             X.append(X_i)
#             y.append(y_i)
#             lens.append(len(modified_line_i.split()))

#         output.append((input_lines, modified_lines, np.array(X), np.array(y), lens))
#     return output

############################################################

def _get_line_representation(line, rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25]):
    modified_words = []
    for word in line.split():
        rep_type = np.random.choice(rep_list, 1, p=probs)[0]
        if 'swap' in rep_type:
            # word_rep, new_word = get_swap_word_representation(word)
            new_word = get_swap_word_representation(word)
        elif 'drop' in rep_type:
            # word_rep, new_word = get_drop_word_representation(word, 1.0)
            new_word = get_drop_word_representation(word, 1.0)
        elif 'add' in rep_type:
            # word_rep, new_word = get_add_word_representation(word)
            new_word = get_add_word_representation(word)
        elif 'key' in rep_type:
            # word_rep, new_word = get_keyboard_word_representation(word)
            new_word = get_keyboard_word_representation(word)
        elif 'none' in rep_type or 'normal' in rep_type:
            # word_rep, _ = get_swap_word_representation(word)
            # new_word = word
            new_word = word
        else:
            #TODO: give a more ceremonious error...
            raise NotImplementedError
        # rep.append(word_rep)
        modified_words.append(new_word)
    # return rep, " ".join(modified_words)
    return " ".join(modified_words)

def get_line_representation(lines, rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25]):
    # rep = []
    modified_lines = [_get_line_representation(line,rep_list,probs) for line in lines]
    return modified_lines


""" 
word representation from individual chars
    one hot (first char) + bag of chars (middle chars) + one hot (last char)
"""
def get_swap_word_representation(word):

    # dirty case
    if len(word) == 1 or len(word) == 2:
        # rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1])
        # return rep, word
        return word

    # rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1])
    if len(word) > 3:
        idx = random.randint(1, len(word)-3)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx+2:]

    # return rep, word
    return word

""" 
word representation from individual chars (except that one of the internal
    chars might be dropped with a probability prob
"""
def get_drop_word_representation(word, prob=0.5):
    p = random.random()
    if len(word) >= 5 and p < prob:
        idx = random.randint(1, len(word)-2)
        word = word[:idx] + word[idx+1:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    elif p > prob:
        # rep, word = get_swap_word_representation(word)
        word = get_swap_word_representation(word)
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word

def get_add_word_representation(word):
    if len(word) >= 3:
        idx = random.randint(1, len(word)-1)
        random_char = _get_random_char()
        word = word[:idx] + random_char + word[idx:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word

def get_keyboard_word_representation(word):
    if len(word) >=3:
        idx = random.randint(1, len(word)-2)
        keyboard_neighbor = _get_keyboard_neighbor(word[idx])
        word = word[:idx] + keyboard_neighbor + word[idx+1:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word


# """ word representation from bag of chars
# """
# def get_boc_word_representation(word):
#     return zero_vector() + bag_of_chars(word) + zero_vector()


# def one_hot(char):
#     # return [1.0 if ch == char else 0.0 for ch in CHAR_VOCAB]
#     return [1.0 if ch == char else 0.0 for ch in "abcdefghijklmnopqrstuvwxyz"]


# def bag_of_chars(chars):
#     # return [float(chars.count(ch)) for ch in CHAR_VOCAB]
#     return [float(chars.count(ch)) for ch in "abcdefghijklmnopqrstuvwxyz"]


# def zero_vector():
#     # return [0.0 for _ in CHAR_VOCAB]
#     return [0.0 for _ in "abcdefghijklmnopqrstuvwxyz"]


#TODO: is that all the characters we need??
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
