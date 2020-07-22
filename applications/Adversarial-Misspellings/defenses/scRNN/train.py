from collections import defaultdict
import numpy as np
import pickle
import random
from random import shuffle
import utils_
from utils_ import * #FIXME: should not do this
import argparse
import time

# torch related imports
import torch
from torch import nn
from torch.autograd import Variable

# elmo related imports
from allennlp.modules.elmo import batch_to_ids

# model related imports
from model import ScRNN
from model import ElmoScRNN
from model import ElmoRNN


parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train-rep', dest='train_rep_list', nargs='+', default=[],
        help = 'the type of the representation to train from')

parser.add_argument('--val-rep', dest='val_rep_list', nargs='+', default=[],
        help = 'the type of the representation to validate on')

parser.add_argument('--train-rep-probs', dest='train_rep_probs', nargs='+', default=[],
        help = 'the probs of the representation to train from')

parser.add_argument('--val-rep-probs', dest='val_rep_probs', nargs='+', default=[],
        help = 'the probs of the representation to validate on')

parser.add_argument('--save', dest='save_model', action='store_true')
parser.add_argument('--background', dest='background', action='store_true')
parser.add_argument('--background-train', dest='background_train', action='store_true')

parser.add_argument('--new-vocab', dest='new_vocab', action='store_true')
parser.add_argument('--model-type', dest='model_type', type=str, default="scrnn",
        help="choice between scrnn/elmo/elmo-plus-scrnn")
parser.add_argument('--model-type-bg', dest='model_type_bg', type=str, default="scrnn",
        help="choice between scrnn/elmo/elmo-plus-scrnn")

parser.add_argument('--no-train', dest='need_to_train', action='store_false')

parser.add_argument('--model-path', dest='model_path', type=str, default="")
parser.add_argument('--model-path-bg', dest='model_path_bg', type=str, default="")

parser.add_argument('--unk-output', dest='unk_output', action='store_true')

parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100)
parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=9999)
parser.add_argument('--vocab-size-bg', dest='vocab_size_bg', type=int, default=78470)

# char vocab path for training bg model to share vocab
parser.add_argument('--common-cv-path', dest='common_cv_path', type=str,
        default="vocab/CHAR_VOCAB_ 16580.p")

# train/dev/test files
parser.add_argument('--train-file', dest='train_file', type=str,
        default="../../../data/classes/train.txt")
parser.add_argument('--dev-file', dest='dev_file', type=str,
        default="../../../data/classes/dev.txt")
parser.add_argument('--test-file', dest='test_file', type=str,
        default="../../../data/classes/test.txt")

parser.add_argument('--task-name', dest='task_name', type=str,
        default="")

params = vars(parser.parse_args())

# useful variables for representation type and strength...
train_rep_list = params['train_rep_list']
val_rep_list = params['val_rep_list']
train_rep_probs = [float(i) for i in params['train_rep_probs']]
val_rep_probs = [float(i) for i in params['val_rep_probs']]
batch_size = params['batch_size']
model_type = params['model_type'].lower()
model_type_bg = params['model_type_bg'].lower()
vocab_size = params['vocab_size']
vocab_size_bg = params['vocab_size_bg']
task_name = params['task_name']
NUM_EPOCHS = params['num_epochs']
set_word_limit(vocab_size, task_name)
WORD_LIMIT = vocab_size
STOP_AFTER = 25

# shall we save the model?
save = params['save_model']

# are we also using a background model?
use_background = params['background']

# are we training the background model?
background_train = params['background_train']


# paths to important stuff..
PWD = "/home/danish/git/break-it-build-it/src/defenses/scRNN/"

# path to vocabs
w2i_PATH = PWD + "vocab/" + task_name  + "w2i_" + str(vocab_size) + ".p"
i2w_PATH = PWD + "vocab/" + task_name + "i2w_" + str(vocab_size) + ".p"
CHAR_VOCAB_PATH = PWD + "vocab/" + task_name + "CHAR_VOCAB_ " + str(vocab_size) + ".p"
common_cv_path = params['common_cv_path']

# paths to background vocabs
w2i_PATH_BG = PWD + "vocab/" + task_name  + "w2i_" + str(vocab_size_bg) + ".p"
i2w_PATH_BG = PWD + "vocab/" + task_name + "i2w_" + str(vocab_size_bg) + ".p"
CHAR_VOCAB_PATH_BG = PWD + "vocab/" + task_name + "CHAR_VOCAB_ " + str(vocab_size_bg) + ".p"

# model paths
MODEL_PATH = PWD + params['model_path']
MODEL_PATH_BG = PWD + params['model_path_bg']

# train/dev/test files
train_file = params['train_file']
dev_file = params['dev_file']
test_file = params['test_file']

# sanity check...
print ("--- Parameters ----")
print (params)

"""
[Takes in predictions (y_preds) in integers, outputs a human readable
output line. In case when the prediction is UNK, it uses the input word as is.
Hence, input_line is also needed to know the corresponding input word.]
"""
def decode_line(input_line, y_preds, use_background, y_preds_bg):
    SEQ_LEN = len(input_line.split())
    assert (SEQ_LEN == len(y_preds))

    predicted_words = []
    for idx in range(SEQ_LEN):
        if y_preds[idx] == WORD_LIMIT:
            word = input_line.split()[idx]
            if use_background:
                # the main model predicted unk ...backoff
                if y_preds_bg[idx] != vocab_size_bg:
                    # the backoff model predicted non-unk
                    word = utils_.i2w_bg[y_preds_bg[idx]]
					# print ("Input: %s \n Backoff: %s -> %s\n" %(input_line, input_line.split()[idx], word))
            if params['unk_output']:
                word = "a"
        else:
            word = utils_.i2w[y_preds[idx]]
        predicted_words.append(word)

    return " ".join(predicted_words)


"""
    [computes the word error rate]
    true_lines are what the model should have predicted, whereas
    output_lines are what the model ended up predicted
"""
def compute_WER(true_lines, output_lines):
    assert (len(true_lines) == len(output_lines))
    size = len(output_lines)

    error = 0.0
    total_words = 0.0

    for i in range(size):
        true_words = true_lines[i].split()
        output_words = output_lines[i].split()
        assert (len(true_words) == len(output_words))
        total_words += len(true_words)
        for j in range(len(output_words)):
            if true_words[j] != output_words[j]:
                error += 1.0

    return (100. * error/total_words)



def iterate(model, optimizer, data_lines, need_to_train, rep_list, rep_probs,
        desc, iter_count, print_stuff=True, use_background=False, model_bg=None):
    data_lines = sorted(data_lines, key = lambda x:len(x.split()), reverse=True)
    Xtype = torch.FloatTensor
    ytype = torch.LongTensor
    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=TARGET_PAD_IDX)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        Xtype = torch.cuda.FloatTensor
        ytype = torch.cuda.LongTensor
        criterion.cuda()

    predicted_lines = []
    true_lines = []

    total_loss = 0.0

    for input_lines, modified_lines, X, y, lens in get_batched_input_data(data_lines, batch_size, \
            rep_list, rep_probs):
        true_lines.extend(input_lines)
        tx = Variable(torch.from_numpy(X)).type(Xtype)
        ty_true = Variable(torch.from_numpy(y)).type(ytype)

        tokenized_modified_lines = [line.split() for line in modified_lines]
        if 'elmo' in model_type or 'elmo' in model_type_bg:
            tx_elmo = Variable(batch_to_ids(tokenized_modified_lines)).type(ytype)

        # forward pass
        if model_type == 'elmo':
            ty_pred = model(tx_elmo)
            #TODO: add the cases where the background model might
            # be other than an elmo-only model
            if use_background and model_type_bg =='elmo':
                ty_pred_bg = model_bg(tx_elmo)
        elif model_type == 'scrnn':
            ty_pred = model(tx, lens)
            #TODO: add the cases where the background model might
            # be other than an scrnn-only model
            if use_background and model_type_bg == 'scrnn':
                ty_pred_bg = model_bg(tx, lens)
        elif 'elmo' in model_type and 'scrnn' in model_type:
            ty_pred = model(tx, tx_elmo, lens)

            #TODO: add the cases where the background model might
            # be an elmo-only model
            if use_background and 'elmo' in model_type_bg and \
            'scrnn' in model_type_bg:
                ty_pred_bg = model_bg(tx, tx_elmo, lens)
            elif use_background and model_type_bg == 'scrnn':
                ty_pred_bg = model_bg(tx, lens)

        y_pred = ty_pred.detach().cpu().numpy()
        # ypred BATCH_SIZE x NUM_CLASSES x SEQ_LEN
        if use_background:
            y_pred_bg = ty_pred_bg.detach().cpu().numpy()


        for idx in range(batch_size):
            y_pred_i = [np.argmax(y_pred[idx][:, i]) for i in range(lens[idx])]

            y_pred_bg_i = None
            if use_background:
                y_pred_bg_i = [np.argmax(y_pred_bg[idx][:, i]) for i in range(lens[idx])]

            predicted_lines.append(decode_line(modified_lines[idx], y_pred_i, use_background, y_pred_bg_i))


        # compute loss
        loss = criterion(ty_pred, ty_true)
        total_loss += loss.item()

        if need_to_train:
            # backprop the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    WER = compute_WER(true_lines, predicted_lines)

    if print_stuff:
        print ("Average %s loss after %d iteration = %0.4f" %(desc, iter_count,
                    total_loss/len(true_lines)))
        print ("Total %s WER after %d iteration = %0.4f" %(desc, iter_count, WER))


    return WER


def main():

    train_lines = get_lines(train_file)

    if params['new_vocab']:
        print ("creating new vocabulary")
        create_vocab(train_file, background_train, common_cv_path)
    else:
        print ("loading existing vocabulary")
        load_vocab_dicts(w2i_PATH, i2w_PATH, CHAR_VOCAB_PATH)
        if use_background:
            print ("loading existing background vocabulary")
            load_vocab_dicts(w2i_PATH_BG, i2w_PATH_BG, CHAR_VOCAB_PATH_BG, use_background)


    print ("len of w2i ", len(utils_.w2i))
    print ("len of i2w ", len(utils_.i2w))
    print ("len of char vocab", len(utils_.CHAR_VOCAB))

    if params['need_to_train']:
        print ("Word limit from utils_ ", WORD_LIMIT)
        if model_type == 'elmo':
            print ("Initializing an only elmo model")
            model = ElmoRNN(WORD_LIMIT + 1) # +1 for UNK
        elif model_type == 'scrnn':
            print ("Initializing an only ScRNN model")
            model = ScRNN(len(utils_.CHAR_VOCAB), 50, WORD_LIMIT + 1) # +1 for UNK
        elif 'elmo' in model_type and 'scrnn' in model_type:
            print ("Initializing a ScRNN plus Elmo model")
            model = ElmoScRNN(len(utils_.CHAR_VOCAB), 50, WORD_LIMIT + 1) # +1 for UNK
    else:
        model = torch.load(MODEL_PATH)
        model_bg = None
        if use_background:
            print ("Loading background model")
            model_bg = torch.load(MODEL_PATH_BG)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=TARGET_PAD_IDX)
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        model.cuda()
        if use_background:
            model_bg.cuda()

    val_lines = get_lines(dev_file)
    test_lines = get_lines(test_file)

    if params['need_to_train']:
        # begin training ...
        print (" *** training the model *** ")
        best_val_WER = 100.0
        last_dumped_idx = 99999
        for ITER in range(NUM_EPOCHS):
            st_time = time.time()
            _ = iterate(model, optimizer, train_lines, True, train_rep_list,
                    train_rep_probs, 'train', ITER+1)

            curr_val_WER = iterate(model, None, val_lines, False, val_rep_list,
                    val_rep_probs, 'val', ITER+1)

            _ = iterate(model, None, test_lines, False, val_rep_list,
                    val_rep_probs, 'test', ITER+1)


            if save:
                # check if the val WER improved?
                if curr_val_WER < best_val_WER:
                    last_dumped_idx = ITER+1
                    best_val_WER = curr_val_WER
                    # informative names for model dump files
                    train_rep_names = "_".join(train_rep_list)
                    train_probs_names = ":".join([str(i) for i in train_rep_probs])

                    print ("Dumping after ", ITER + 1)
                    model_name = model_type
                    torch.save(model,
                            "tmp/consideration/" + model_name +  \
                            "_TASK_NAME=" + task_name + \
                            "_VOCAB_SIZE=" + str(vocab_size) + \
                            "_REP_LIST=" + train_rep_names + \
                            "_REP_PROBS=" + train_probs_names)

            # report the time taken per iteration for train + val + test
            # (+ often save)
            en_time = time.time()
            print ("Time for the iteration %0.1f seconds" %(en_time - st_time))

            # check if there hasn't been enough progress since last few iters
            if ITER > STOP_AFTER + last_dumped_idx:
                # i.e it is not improving since 'STOP_AFTER' number of iterations
                print ("Aborting since there hasn't been much progress")
                break

    else:
        # just run the model on validation and test...
#print (" *** running the model on val and test set *** ")

        st_time = time.time()

        val_WER = iterate(model, None, val_lines, False, val_rep_list,
                val_rep_probs, 'val', 0, use_background=use_background, model_bg=model_bg)

        test_WER = iterate(model, None, test_lines, False, val_rep_list,
                val_rep_probs, 'test', 0, False, use_background=use_background, model_bg=model_bg)

        # report the time taken per iteration for val + test
        en_time = time.time()
        print ("Time for the testing process = %0.1f seconds" %(en_time - st_time))
        val_rep_names = " ".join(val_rep_list)
        model_name = MODEL_PATH.split("/")[-1]
        print (val_rep_names + "\t" + model_name + "\t" + str(val_WER) + "\t"\
                + str(test_WER))


    return


main()
