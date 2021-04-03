# Usage: python3 main.py --mode dev --model bilstm \
# --load tmp/lstm_epochs=4 --num_examples 100
from collections import defaultdict
import time
import datetime
import biLstm_with_chars # word + char model biLSTM model
import biLstm_char_only  # char only biLSTM model
import biLstm
#from biLstm_with_chars import BiLSTM
# from CNN import CNN
# from RNN_with_char import RNN
import argparse
from random import shuffle
import sys
import string
from nltk.corpus import stopwords as SW
#import hunspell_checker
#from hunspell_checker import HunspellChecker
import attacks
from tqdm import tqdm
import pickle

import dynet_config
dynet_config.set(random_seed=42)
import dynet as dy

import numpy as np
np.random.seed(42)
import random
random.seed(42)

sys.path.insert(0, 'defenses/scRNN/')
sys.path.append('spell_checkers/')
from spell_checkers.atd_checker import ATDChecker
from corrector import ScRNNChecker
from neuspell import CorrectorElmoSCLstm
# personal logging lib
import log
log.DEBUG = True


stopwords = set(SW.words("english")) | set(string.punctuation)
# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
s2i = defaultdict(lambda: len(s2i))
c2i = defaultdict(lambda: len(c2i))

UNK = w2i["<unk>"]
CHAR_UNK = c2i["<unk>"]
NUM_EXAMPLES = 100

vocab_set = set()
char_vocab_set = set()


def read_valid_lines(filename):
    """reads files (ignores the neutral reviews)

    Arguments:
        filename -- data file

    Returns:
        lines, tags: list of reviews, and their tags
    """

    print("starting to read %s" %(filename))

    lines, tags = [], []
    with open(filename, 'r') as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            if tag == '0' or tag == '1': tag = '0'
            if tag == '3' or tag == '4': tag = '1'
            if tag == '2': continue
            tags.append(tag)
            lines.append(words)
    return lines, tags


def read_dataset(filename, drop=False, swap=False, key=False, add=False, all=False):
    """creates a dataset from reading reviews; uses word and tag dicts

    Arguments:
        filename  -- input file
    """

    lines, tags = read_valid_lines(filename)
    ans = []
    for line, tag in zip(lines, tags):
        words = [x for x in line.split(" ")]
        word_idxs = [w2i[x] for x in line.split(" ")]
        char_idxs = []
        for word in words: char_idxs.append([c2i[i] for i in word])
        tag = t2i[tag]
        ans.append((word_idxs, char_idxs, tag))
        if (drop or swap or key or add or all) and random.random() < char_drop_prob:
            if drop:
                line = drop_a_char(line)
            elif swap:
                line = swap_a_char(line)
            elif key:
                line = key_a_char(line)
            elif add:
                line = add_a_char(line)
            elif all:
                perturbation_fns = [drop_a_char, swap_a_char, add_a_char, swap_a_char]
                perturbation_fn = np.random.choice(perturbation_fns, 1)[0]
                line = perturbation_fn(line)

            words = [x for x in line.split(" ")]
            word_idxs = [w2i[x] for x in line.split(" ")]
            char_idxs = []
            for word in words: char_idxs.append([c2i[i] for i in word])
            ans.append((word_idxs, char_idxs, tag))
    return ans



def normalize(x):
    """ normalizes the scores in x, works only for 1D """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def create_vocabulary(filename):
    """given a file, creates the vocab set from its words

    Arguments:
        filename -- input file
    """

    global vocab_set
    lines, _ = read_valid_lines(filename)
    for line in lines:
        for word in line.split(" "):
            vocab_set.add(word)
            for char in word:
                char_vocab_set.add(char)
    return


def get_word_and_char_indices(line):
    words = [x for x in line.split(" ")]
    word_idxs = [w2i[x] for x in line.split(" ")]
    char_idxs = []
    for word in words: char_idxs.append([c2i[i] for i in word])
    return word_idxs, char_idxs


def check_against_spell_mistakes(filename):
    lines, tags = read_valid_lines(filename)

    c = list(zip(lines, tags))
    random.shuffle(c)
    lines, tags = zip(*c)
    lines = lines
    tags = tags

    # if in small (or COMPUTATION HEAVY) modes
    if params['small']:
        lines = lines[:200]
        tags = tags[:200]
    if params['small'] and params['sc_atd']:
        lines = lines[:99]
        tags = tags[:99]

    inc_count = 0.0
    inc_count_per_attack = [0.0 for _ in range(NUM_ATTACKS+1)]
    error_analyser = {}
    for line, tag in tqdm(zip(lines, tags)):

        w_i, c_i = get_word_and_char_indices(line)
        if params['is_spell_check']:
            w_i, c_i = get_word_and_char_indices(checker.correct_string(line))

        # check if model prediction is incorrect, if yes, continue
        model_prediction = predict(w_i, c_i)
        if t2i[tag] != model_prediction:
            # already incorrect, no attack needed
            inc_count += 1
            inc_count_per_attack[0] += 1.0
            continue

        found_incorrect = False

        worst_example = line
        worst_confidence = 1.0
        worst_idx = -1
        ignore_incides=set()

        for attack_count in range(1, 1 + NUM_ATTACKS):

            ignore_incides.add(worst_idx)

            if 'drop' in type_of_attack:
                gen_attacks = attacks.drop_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
            elif 'swap' in type_of_attack:
                gen_attacks = attacks.swap_one_attack(worst_example, include_ends=params['include_ends'])
            elif 'key' in type_of_attack:
                gen_attacks = attacks.key_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
            elif 'add' in type_of_attack:
                gen_attacks = attacks.add_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])
            elif 'all' in type_of_attack:
                gen_attacks = attacks.all_one_attack(worst_example, ignore_incides, include_ends=params['include_ends'])

            for idx, adversary in gen_attacks:
                original_adv = adversary
                if found_incorrect: break
                if params['is_spell_check']:
                    adversary = checker.correct_string(adversary)
                w_i, c_i = get_word_and_char_indices(adversary)
                adv_pred = predict(w_i, c_i)
                confidence = get_confidence(w_i, c_i)

                if confidence < worst_confidence:
                    worst_confidence = confidence
                    worst_idx = idx
                    worst_example = adversary

                if adv_pred != t2i[tag]:
                    # found incorrect prediction
                    found_incorrect = True
                    break

            if found_incorrect:
                inc_count += 1.0
                inc_count_per_attack[attack_count] += 1.0
                if params['analyse']:
                    error_analyser[line] = {}
                    error_analyser[line]['adversary'] = original_adv.split()[idx]
                    error_analyser[line]['correction'] = adversary.split()[idx]
                    error_analyser[line]['idx'] = idx

                break

    for num in range(NUM_ATTACKS + 1):
        log.pr_red('adversarial accuracy of the model after %d attacks = %.2f'
                %(num, 100. * (1 - sum(inc_count_per_attack[:num+1])/len(lines))))

    if params['analyse']:
        curr_time = datetime.datetime.now().strftime("%B_%d_%I:%M%p")
        pickle.dump(error_analyser, open("error_analyser_" + str(curr_time) + ".p", 'wb'))

    return None


# make argparse
parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load', dest='input_file', type=str, default="",
        help = 'load already trained model')
parser.add_argument('--save', dest='output_file', type=str, default="",
        help = 'save existing model')
parser.add_argument('--model', dest='model_type', type=str, default="lstm",
        help = 'architecture of the model: lstm or rnn or cnn')
parser.add_argument('--mode', dest='mode', type=str, default="dev",
        help = 'training or dev?')
parser.add_argument('--attack', dest='type_of_attack', type=str, default=None,
        help='type of attack you want, swap/drop/add/key/all')
parser.add_argument('--small', dest='small', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--backoff', dest='backoff_mode', type=str, default="",
        help = 'neutral or pass-through')

parser.add_argument('--defense', dest='is_spell_check', action='store_true')
parser.add_argument('--sc-neutral', dest='unk_output', action='store_true')
parser.add_argument('--sc-background', dest='sc_background', action='store_true')
parser.add_argument('--analyse', dest='analyse', action='store_true')
parser.set_defaults(is_spell_check=False)

parser.add_argument('--include-ends', dest='include_ends', action='store_true')
parser.set_defaults(include_ends=False)

# data augmentation flags
parser.add_argument('--da-drop', dest='da_drop', action='store_true')
parser.add_argument('--da-key', dest='da_key', action='store_true')
parser.add_argument('--da-add', dest='da_add', action='store_true')
parser.add_argument('--da-swap', dest='da_swap', action='store_true')
parser.add_argument('--da-all', dest='da_all', action='store_true')
parser.add_argument('--da-drop-prob', dest='da_drop_prob', type=float, default=0.5)

parser.add_argument('--num-attacks', dest='num_attacks', type=int, default=0)
parser.add_argument('--dynet-seed', dest='dynet-seed', type=int, default=42)

# adversarial training flags
parser.add_argument('--adv-drop', dest='adv_drop', action='store_true')
parser.add_argument('--adv-swap', dest='adv_swap', action='store_true')
parser.add_argument('--adv-key', dest='adv_key', action='store_true')
parser.add_argument('--adv-add', dest='adv_add', action='store_true')
parser.add_argument('--adv-all', dest='adv_all', action='store_true')
parser.add_argument('--adv-prob', dest='adv_prob', type=float, default=0.1)

# model names for spell check models
parser.add_argument('--sc-model-path', dest='sc_model_path', type=str, default=None,
        help = 'the model path for ScRNN model')
parser.add_argument('--sc-model-path-bg', dest='sc_model_path_bg', type=str, default=None,
        help = 'the model path for ScRNN background model')
parser.add_argument('--sc-elmo', dest='sc_elmo', action='store_true')
parser.add_argument('--sc-elmo-bg', dest='sc_elmo_bg', action='store_true')
parser.add_argument('--sc-atd', dest='sc_atd', action='store_true')
parser.add_argument('--sc-elmoscrnn', dest='sc_elmoscrnn', action='store_true')
parser.add_argument('--sc-vocab-size', dest='sc_vocab_size', type=int, default=9999)
parser.add_argument('--sc-vocab-size-bg', dest='sc_vocab_size_bg', type=int, default=78470)

parser.add_argument('--task-name', dest='task_name', type=str, default="")

params = vars(parser.parse_args())

# logging details
log.DEBUG = params['debug']

model_type = params['model_type']
input_file = params['input_file']
mode = params['mode']
type_of_attack = params['type_of_attack']
char_drop_prob = params['da_drop_prob']
NUM_ATTACKS = params['num_attacks']

SC_MODEL_PATH = params['sc_model_path']
SC_MODEL_PATH_BG = params['sc_model_path_bg']

if params['sc_atd']:
    checker = ATDChecker()
elif params['sc_elmoscrnn']:
    print("###########")
    print("using new spell corrector")
    print(f"using backoff={params['backoff_mode']}")
    print("###########")
    checker = CorrectorElmoSCLstm(DATA_FOLDER_PATH=ELMOSCRNN_DATA_FOLDER_PATH, backoff=params['backoff_mode'])
elif SC_MODEL_PATH_BG is None or params['sc_background']:
    # only foreground spell correct model...
    checker = ScRNNChecker(model_name=SC_MODEL_PATH, use_background=False,
            unk_output=params['unk_output'], use_elmo=params['sc_elmo'],
            task_name=params['task_name'], vocab_size=params['sc_vocab_size'],
            vocab_size_bg=params['sc_vocab_size_bg'])
else:
    checker = ScRNNChecker(model_name=SC_MODEL_PATH, model_name_bg=SC_MODEL_PATH_BG,
            use_background=True, unk_output=params['unk_output'],
            use_elmo=params['sc_elmo'], use_elmo_bg=params['sc_elmo_bg'],
            task_name=params['task_name'], vocab_size=params['sc_vocab_size'],
            vocab_size_bg=params['sc_vocab_size_bg'])

model = None
train = read_dataset("data/classes/train.txt")

# modify the dicts so that they return unk for unseen words/chars
w2i = defaultdict(lambda: UNK, w2i)
c2i = defaultdict(lambda: CHAR_UNK, c2i)

dev = read_dataset("data/classes/dev.txt")
test = read_dataset("data/classes/test.txt")


def evaluate(filename="data/classes/test.txt"):
    lines, tags = read_valid_lines(filename)
    correct = 0.0
    for line, tag in tqdm(zip(lines, tags)):
        w_i, c_i = get_word_and_char_indices(line)
        pred = predict(w_i, c_i)
        if pred == t2i[tag]: correct += 1.0
    log.pr_green("accuracy of the model on test set = %.4f [No spell checks]" % (correct / len(lines)))
    return


def predict(words, chars):
    scores = model.calc_scores(words, chars)
    pred = np.argmax(scores.npvalue())
    return pred


def get_confidence(words, chars):
    scores = model.calc_scores(words, chars)
    normalized_scores = normalize(scores.npvalue())
    pred = np.argmax(scores.npvalue())
    return normalized_scores[pred]


def drop_a_char(sentence):
    words = sentence.split(" ")

    for _ in range(10):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) < 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def swap_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-3)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        words[word_idx] = words[word_idx][:pos] + words[word_idx][pos:pos+2][::-1] + words[word_idx][pos+2:]
        sentence = " ".join(words)
        break
    return sentence

def key_a_char(sentence):
    words = sentence.split(" ")
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-2)
        neighboring_chars = attacks.get_keyboard_neighbors(words[word_idx][pos])
        random_neighbor = np.random.choice(neighboring_chars, 1)[0]
        words[word_idx] = words[word_idx][:pos] + random_neighbor + words[word_idx][pos+1:]
        sentence = " ".join(words)
        break
    return sentence

def add_a_char(sentence):
    words = sentence.split(" ")
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    for _ in range(100):
        word_idx = random.randint(0, len(words)-1)
        if len(words[word_idx]) <= 3: continue
        pos = random.randint(1, len(words[word_idx])-1)
        #words[word_idx] = words[word_idx][:pos] + words[word_idx][pos+1:]
        new_char = np.random.choice(alphabets, 1)[0]
        words[word_idx] = words[word_idx][:pos] + new_char + words[word_idx][pos:]
        sentence = " ".join(words)
        break
    return sentence


def start_adversarial_training(trainer):
    lines, tags = read_valid_lines("data/classes/train.txt")
    train = [(lines[i], tags[i]) for i in range(len(lines))]
    for ITER in range(10):
        train_loss = 0.0
        train_correct = 0.0
        start = time.time()
        #TODO shuffle train
        random.shuffle(train)
        print("Length of training examples = %d" %(len(train)))
        for line, tag in train:
            w_i, c_i = get_word_and_char_indices(line)
            scores = model.calc_scores(w_i, c_i)
            my_loss = dy.pickneglogsoftmax(scores, t2i[tag])
            train_loss += my_loss.value()
            my_loss.backward()
            trainer.update()
            pred = np.argmax(scores.npvalue())
            if pred == t2i[tag]: train_correct += 1
        print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
        print("iter %r: train acc=%.4f" % (ITER, train_correct / len(train)))
        # Compute dev loss
        dev_loss = 0.0
        dev_correct = 0.0
        for words, chars, tag in dev:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            dev_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: dev_correct += 1
        print("iter %r: dev loss/sent=%.4f, time=%.2fs" % (ITER, dev_loss / len(dev), time.time() - start))
        print("iter %r: dev acc=%.4f" % (ITER, dev_correct / len(dev)))

        # compute test loss
        test_loss = 0.0
        test_correct = 0.0
        for words, chars, tag in test:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            test_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: test_correct += 1
        print("iter %r: test loss/sent=%.4f, time=%.2fs" % (ITER, test_loss / len(test), time.time() - start))
        print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        model.save("tmp/adv-" + model_type + "_drop=" + str(params['adv_drop']) + "_swap=" + str(params['adv_swap'])  + "_key" + str(params['adv_key']) + "_add" + str(params['adv_add']) + "_all" + str(params['adv_all']) + "_prob=" +  str(params['adv_prob']) +  "_epochs=" + str(ITER))

        if params['adv_swap'] or params['adv_drop'] or params['adv_key'] or params['adv_add'] or params['adv_all']:
            train.extend(add_more_examples(train,  params['adv_prob']/(ITER+2),
                        drop=params['adv_drop'], swap=params['adv_swap'],
                        key=params['adv_key'], add=params['adv_add'],
                        all=params['adv_all']))


def start_training(train, dev, trainer):
    if params['da_drop']:
        train = read_dataset("data/classes/train.txt", drop=True)
    if params['da_swap']:
        train = read_dataset("data/classes/train.txt", swap=True)
    if params['da_key']:
        train = read_dataset("data/classes/train.txt", key=True)
    if params['da_add']:
        train = read_dataset("data/classes/train.txt", add=True)
    if params['da_all']:
        train = read_dataset("data/classes/train.txt", all=True)
    for ITER in range(10):
        # Perform training
        random.shuffle(train)
        train_loss = 0.0
        start = time.time()
        train_correct = 0.0
        for words, chars, tag in train:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            train_loss += my_loss.value()
            my_loss.backward()
            trainer.update()
            pred = np.argmax(scores.npvalue())
            if pred == tag: train_correct += 1
        print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
        print("iter %r: train acc=%.4f" % (ITER, train_correct / len(train)))
        # Compute dev loss
        dev_loss = 0.0
        dev_correct = 0.0
        for words, chars, tag in dev:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            dev_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: dev_correct += 1
        print("iter %r: dev loss/sent=%.4f, time=%.2fs" % (ITER, dev_loss / len(dev), time.time() - start))
        print("iter %r: dev acc=%.4f" % (ITER, dev_correct / len(dev)))

        # compute test loss
        test_loss = 0.0
        test_correct = 0.0
        for words, chars, tag in test:
            scores = model.calc_scores(words, chars)
            my_loss = dy.pickneglogsoftmax(scores, tag)
            test_loss += my_loss.value()
            pred = np.argmax(scores.npvalue())
            if pred == tag: test_correct += 1
        print("iter %r: test loss/sent=%.4f, time=%.2fs" % (ITER, test_loss / len(test), time.time() - start))
        print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test)))
        model.save("tmp/" + model_type + "_drop=" + str(params['da_drop']) + "_swap=" + str(params['da_swap']) + "_key=" + str(params['da_key']) + "_add=" + str(params['da_add']) + "_all=" + str(params['da_all'])  + "_prob=" +  str(char_drop_prob) +  "_epochs=" + str(ITER))


def get_qualitative_examples():
    lines, tags = read_valid_lines("data/classes/test.txt")
    c = list(zip(lines, tags))
    random.shuffle(c)
    lines, tags = zip(*c)
    lines = lines[:200]
    tags = tags[:200]

    for line, tag in tqdm(zip(lines, tags)):

        w_i, c_i = get_word_and_char_indices(line)

        # check if model prediction is incorrect, if yes, find next example...
        model_prediction = predict(w_i, c_i)
        if t2i[tag] != model_prediction:
            # already incorrect, not interesting...
            continue

        gen_attacks = attacks.all_one_attack(line)

        for idx, adversary in gen_attacks:
                #adversary = checker.correct_string(adversary)
            w_i, c_i = get_word_and_char_indices(adversary)

            adv_pred = predict(w_i, c_i)

            if adv_pred == t2i[tag]:
                # this example doesn't break the model...
                continue

            corrected_string = checker.correct_string(adversary)
            w_i, c_i = get_word_and_char_indices(corrected_string)

            post_pred = predict(w_i, c_i)

            if post_pred != t2i[tag]:
                # after correction the tag isn't correct...
                continue

            log.pr(" -------------- ")
            log.pr("Original line = %s" %(line))
            log.pr("Original label = %s" %(tag))
            log.pr_red ("Adversary = %s" %(adversary))
            log.pr_green("Correction = %s" %(corrected_string))
            log.pr(" -------------- ")

    return None


def generate_ann():
    generate_dict = dict()
    lines, tags = read_valid_lines("data/classes/test.txt")
    c = list(zip(lines, tags))
    random.shuffle(c)
    lines, tags = zip(*c)
    lines = lines[:200]
    tags = tags[:200]

    # get the missclassified ones first
    missclassified_count = 0

    final_list = []

    for idx, line in enumerate(lines):
        if missclassified_count >= 50: break
        # need to attack the line...
        for _, adv in attacks.random_all_one_attack(line):
            w_i, c_i = get_word_and_char_indices(adv)
            model_prediction = predict(w_i, c_i)
            if model_prediction != t2i[tags[idx]]:
                # adversary found...
                final_list.append((line, tags[idx], 1))
                missclassified_count += 1


    for i in range(idx+1, idx+1+50):
        final_list.append((lines[i], tags[i], 0))

    pickle.dump(final_list, open("final_list_annotations.p, 'wb'"))

    for l in final_list:
        print (l[0].strip() + "\t" + str(l[1]) + "\t" + str(l[2]))


def add_more_examples(train, prob=0.1, drop=False, swap=False, key=False, add=False, all=False):
    extra_examples = []
    for line, tag in train:
        if random.random() > prob: continue # this is correct...
        if swap:
            gen_attacks = attacks.swap_one_attack(line)
        elif drop:
            gen_attacks = attacks.drop_one_attack(line)
        elif add:
            gen_attacks = attacks.add_one_attack(line)
        elif key:
            gen_attacks = attacks.key_one_attack(line)
        elif all:
            gen_attacks = attacks.all_one_attack(line)


        for _, adversary in gen_attacks:
                w_i, c_i = get_word_and_char_indices(adversary)
                adv_pred = predict(w_i, c_i)
                if adv_pred != t2i[tag]:
                    # found incorrect
                    extra_examples.append((adversary, tag))

    return extra_examples


def decode_tag(tag):
    return "POSITIVE" if tag == t2i['1'] else "NEGATIVE"


def main():
    # Read in the data
    global model

    nwords = len(w2i)
    ntags = len(t2i)
    nchars = len(c2i)

    if 'rnn' in model_type.lower():
        print ("Running a RNN model")
        model = RNN()
    elif 'cnn' in model_type.lower():
        print ("Running a CNN model")
        model = CNN()
    elif 'bilstm' == model_type.lower():
        print ("Running a BiLSTM char + word model ")
        model = biLstm_with_chars.BiLSTM()
    elif 'bilstm' in  model_type.lower() and 'word' in model_type.lower():
        print ("Running a BiLSTM word only model ")
        model = biLstm.BiLSTM()
    elif 'bilstm' in  model_type.lower() and 'char' in model_type.lower():
        print ("Running a BiLSTM char only model ")
        model = biLstm_char_only.BiLSTM()

    print ("building vocabulary...")
    create_vocabulary('data/classes/train.txt')
    print ("done building vocabulary...")
    print ('size of the character vocab %s' %(len(char_vocab_set)))
    trainer = model.build_model(nwords, nchars, ntags)

    if input_file != "":
        model.load(input_file)

    if 'train' in mode.lower():
        if params['adv_swap'] or params['adv_drop'] or params['adv_key'] \
            or params['adv_add'] or params['adv_all']:
            start_adversarial_training(trainer)
        else:
            start_training(train, dev, trainer)
    elif 'gen' in mode.lower():
        generate_ann()
    elif 'examples' in mode.lower():
        get_qualitative_examples()
    else:
        evaluate()
        if type_of_attack is not None:
            check_against_spell_mistakes('data/classes/test.txt')

main()
