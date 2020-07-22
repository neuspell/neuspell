""" class using Semi Character RNNs as a defense mechanism
    ScRNN paper: https://arxiv.org/abs/1608.02214
"""
import os
import utils_
from utils_ import * #FIXME: this shouldn't be this way

# torch related imports
import torch
from torch import nn
from torch.autograd import Variable

# elmo related imports
from allennlp.modules.elmo import batch_to_ids

class ScRNNChecker(object):
    def __init__(self, model_name=None, model_name_bg=None, vocab_size=9999,\
        vocab_size_bg=78470, use_background=False, unk_output=False, \
        use_elmo=False,  use_elmo_bg=False, task_name=""):
        PWD = os.path.dirname(os.path.realpath(__file__))
        if model_name is None:
            MODEL_PATH = PWD + "/model_dumps/scrnn_VOCAB_SIZE=9999_REP_LIST=swap_key_add_drop_REP_PROBS=0.25:0.25:0.25:0.25"
        else:
            MODEL_PATH = PWD + "/" + model_name

        if use_background and model_name_bg is None:
            MODEL_PATH_BG = PWD + "/model_dumps/scrnn_VOCAB_SIZE=78470_REP_LIST=swap_add_key_drop_REP_PROBS=0.25:0.25:0.25:0.25"
        elif use_background:
            MODEL_PATH_BG = PWD + "/" + model_name_bg


        self.vocab_size_bg = vocab_size_bg
        self.vocab_size = vocab_size
        self.unk_output = unk_output

        # path to vocabs
        w2i_PATH = PWD + "/vocab/" + task_name  + "w2i_" + str(vocab_size) + ".p"
        i2w_PATH = PWD + "/vocab/" + task_name + "i2w_" + str(vocab_size) + ".p"
        CHAR_VOCAB_PATH = PWD + "/vocab/" + task_name + "CHAR_VOCAB_ " + str(vocab_size) + ".p"

        # paths to background vocabs
        w2i_PATH_BG = PWD + "/vocab/" + task_name  + "w2i_" + str(vocab_size_bg) + ".p"
        i2w_PATH_BG = PWD + "/vocab/" + task_name + "i2w_" + str(vocab_size_bg) + ".p"
        CHAR_VOCAB_PATH_BG = PWD + "/vocab/" + task_name + "CHAR_VOCAB_ " + str(vocab_size_bg) + ".p"

        set_word_limit(vocab_size, task_name)

        load_vocab_dicts(w2i_PATH, i2w_PATH, CHAR_VOCAB_PATH)
        if use_background:
            load_vocab_dicts(w2i_PATH_BG, i2w_PATH_BG, CHAR_VOCAB_PATH_BG, use_background)
        self.model = torch.load(MODEL_PATH)
        if use_background:
            self.model_bg = torch.load(MODEL_PATH_BG)
        self.predicted_unks = 0.0
        self.predicted_unks_in_vocab = 0.0
        self.total_predictions = 0.0
        self.use_background = use_background
        self.use_elmo = use_elmo
        self.use_elmo_bg = use_elmo_bg
        return


    def correct_string(self, line):
        line = line.lower()
        Xtype = torch.FloatTensor
        ytype = torch.LongTensor
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.model.cuda()
            Xtype = torch.cuda.FloatTensor
            ytype = torch.cuda.LongTensor
            if self.use_background: self.model_bg.cuda()

        X, _ = get_line_representation(line)
        tx = Variable(torch.from_numpy(np.array([X]))).type(Xtype)

        if self.use_elmo or self.use_elmo_bg:
            tx_elmo = Variable(batch_to_ids([line.split()])).type(ytype)


        SEQ_LEN = len(line.split())

        if self.use_elmo:
            ty_pred = self.model(tx, tx_elmo, [SEQ_LEN])
        else:
            ty_pred = self.model(tx, [SEQ_LEN])

        y_pred = ty_pred.detach().cpu().numpy()
        y_pred = y_pred[0] # ypred now is NUM_CLASSES x SEQ_LEN

        if self.use_background:
            if self.use_elmo_bg:
                ty_pred_bg = self.model_bg(tx, tx_elmo, [SEQ_LEN])
            else:
                ty_pred_bg = self.model_bg(tx, [SEQ_LEN])
            y_pred_bg = ty_pred_bg.detach().cpu().numpy()
            y_pred_bg = y_pred_bg[0]

        output_words = []

        self.total_predictions += SEQ_LEN

        for idx in range(SEQ_LEN):
            pred_idx = np.argmax(y_pred[:, idx])
            if pred_idx == utils_.WORD_LIMIT:
                word = line.split()[idx]
                if self.use_background:
                    pred_idx_bg = np.argmax(y_pred_bg[:, idx])
                    if pred_idx_bg != self.vocab_size_bg:
                        word = utils_.i2w_bg[pred_idx_bg]
                if self.unk_output:
                    word = "a" # choose a sentiment neutral word
                output_words.append(word)
                self.predicted_unks += 1.0
                if word in utils_.w2i:
                    self.predicted_unks_in_vocab += 1.0
            else:
                output_words.append(utils_.i2w[pred_idx])

        return " ".join(output_words)

    def reset_counters(self):
        self.predicted_unks = 0.0
        self.total_predictions = 0.0


    def report_statistics(self):
        print ("Total number of words predicted by background model = %0.2f " %(100. * self.predicted_unks/self.total_predictions))
        print ("Total number of in vocab words predicted by background model = %0.2f " %(100. * self.predicted_unks_in_vocab/self.total_predictions))
