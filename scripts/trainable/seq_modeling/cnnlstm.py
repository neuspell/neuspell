
#############################################
# USAGE
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py probword ../../data -1
#
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py none ../../data 1
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py random ../../data 1
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py word ../../data 1
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py prob ../../data 1
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py probword ../../data 1
# CUDA_VISIBLE_DEVICES=1 python cnnlstm.py probword_v2 ../../data 1
#############################################


############################################
# TO-DO 
# ----
# 1. How to set multip-gpu in torch for training
############################################

import os, sys
# export CUDA_VISIBLE_DEVICES=1,2 && echo $CUDA_VISIBLE_DEVICES
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/.")

from tqdm import tqdm
import os, sys
import numpy as np
import re
import time
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from helpers import progressBar
from helpers import load_vocab_dict, save_vocab_dict
from helpers import load_data, train_validation_split, get_char_tokens, get_tokens, num_unk_tokens
from helpers import batch_iter, labelize, tokenize, char_tokenize
from helpers import untokenize, untokenize_without_unks, untokenize_without_unks2, get_model_nparams
from helpers import batch_accuracy_func

from helpers2 import get_line_representation, get_lines

from models import CharCNNWordLSTMModel

from evals import get_metrics

def load_model(vocab):

    CHAR_EMBS_DIM = 100
    model = CharCNNWordLSTMModel(len(vocab["chartoken2idx"]),
                                 CHAR_EMBS_DIM,
                                 vocab["chartoken2idx"][ vocab["char_pad_token"] ],
                                 vocab["token2idx"][ vocab["pad_token"] ],
                                 len(vocab["token_freq"]))
    print(model)
    print( get_model_nparams(model) )

    return model

def load_pretrained(model, CHECKPOINT_PATH, optimizer=None, device='cuda'):

    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {CHECKPOINT_PATH}")
    checkpoint_data = torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"), map_location=map_location)
    # print(f"previously model saved at : {checkpoint_data['epoch_id']}")

    model.load_state_dict(checkpoint_data['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc
    
    return model

def model_predictions(model, data, vocab, DEVICE, BATCH_SIZE=16):
    """
    model: an instance of CharCNNWordLSTMModel
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    """
    
    topk = 1
    print("###############################################")
    inference_st_time = time.time()
    final_sentences = []
    VALID_BATCH_SIZE = BATCH_SIZE
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    for batch_id, (batch_clean_sentences,batch_corrupt_sentences) in tqdm(enumerate(data_iter)):
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = char_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_==batch_lengths).all()==True
        batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        # forward
        with torch.no_grad():
            # because topk=1, batch_predictions are of shape (batch_size,batch_max_seq_len)
            _, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels, topk=topk)
        batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences)
        final_sentences.extend(batch_predictions)
    print("total inference time for this data is: {:4f} secs".format(time.time()-inference_st_time))
    return final_sentences

def model_inference(model, data, topk, DEVICE, BATCH_SIZE=16, vocab_=None):
    """
    model: an instance of CharCNNWordLSTMModel
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    topk: how many of the topk softmax predictions are considered for metrics calculations
    """
    if vocab_ is not None:
        vocab = vocab_
    print("###############################################")
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_BATCH_SIZE = BATCH_SIZE
    valid_loss = 0.
    valid_acc = 0.
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    for batch_id, (batch_clean_sentences,batch_corrupt_sentences) in tqdm(enumerate(data_iter)):
        st_time = time.time()
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = char_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_==batch_lengths).all()==True
        batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        # forward
        with torch.no_grad():
            """
            NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
            """
            batch_loss, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels, topk=topk)      
        valid_loss += batch_loss
        # compute accuracy in numpy
        batch_labels = batch_labels.cpu().detach().numpy()
        batch_lengths = batch_lengths.cpu().detach().numpy()
        # based on topk, obtain either strings of batch_predictions or list of tokens
        if topk==1:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_corrupt_sentences)    
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_corrupt_sentences, topk=None)

        # batch_clean_sentences = [line.lower() for line in batch_clean_sentences]
        # batch_corrupt_sentences = [line.lower() for line in batch_corrupt_sentences]
        # batch_predictions = [line.lower() for line in batch_predictions]
        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes = \
            get_metrics(batch_clean_sentences,batch_corrupt_sentences,batch_predictions,check_until_topk=topk,return_mistakes=True)
        _corr2corr+=corr2corr
        _corr2incorr+=corr2incorr
        _incorr2corr+=incorr2corr
        _incorr2incorr+=incorr2incorr
        _mistakes.extend(mistakes)
        '''
        # update progress
        progressBar(batch_id+1,
                    int(np.ceil(len(data) / VALID_BATCH_SIZE)), 
                    ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                    [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        '''
    print(f"\nEpoch {None} valid_loss: {valid_loss/(batch_id+1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time()-inference_st_time))
    print("###############################################")
    # print("")
    # for mistake in _mistakes:
    #     print(mistake)
    # print("")
    print("total token count: {}".format(_corr2corr+_corr2incorr+_incorr2corr+_incorr2incorr))
    print(f"_corr2corr:{_corr2corr}, _corr2incorr:{_corr2incorr}, _incorr2corr:{_incorr2corr}, _incorr2incorr:{_incorr2incorr}")
    print(f"accuracy is {(_corr2corr+_incorr2corr)/(_corr2corr+_corr2incorr+_incorr2corr+_incorr2incorr)}")
    print(f"word correction rate is {(_incorr2corr)/(_incorr2corr+_incorr2incorr)}")
    print("###############################################")
    return



























if __name__=="__main__":

    # "word", "prob"
    TRAIN_NOISE_TYPE = sys.argv[1]
    # "../../data"
    BASE_PATH = sys.argv[2]
    # -ve value for inference only; 1 for training a new model from scratch; >1 for continuing training
    START_EPOCH = int(sys.argv[3])
    if START_EPOCH==0:
        raise Exception("START_EPOCH must be a non-zero value")
    # :NEW: finetune now from a specific epoch of a model
    # "probword"
    if len(sys.argv)>4:
        FINETUNE = sys.argv[4]
        if FINETUNE=='probword':
            SRC_CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probwordnoise")
            SRC_VOCAB_PATH = os.path.join(SRC_CHECKPOINT_PATH,"vocab.pkl")
            print(f"Model finetuning with arg: {FINETUNE}, and source model selected from: {SRC_CHECKPOINT_PATH}")
        else:
            raise Exception("only ```probword``` is now supported for finetuning")
        assert os.path.exists(SRC_CHECKPOINT_PATH), print(f"{SRC_CHECKPOINT_PATH} path unavailable")
    else:
        FINETUNE = ""

    #############################################
    # environment
    #############################################    
    
    # checkpoint path for this model
    if TRAIN_NOISE_TYPE=="word":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-wordnoise")
    elif TRAIN_NOISE_TYPE=="prob":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probnoise")
    elif TRAIN_NOISE_TYPE=="random":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-randomnoise")
    elif TRAIN_NOISE_TYPE=="probword":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probwordnoise")
    elif TRAIN_NOISE_TYPE=="probword_v2":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probwordnoise_v2")
    elif TRAIN_NOISE_TYPE=="bea40kfinetune":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probwordnoise-bea40kfinetune")
    elif TRAIN_NOISE_TYPE=="moviereviewsfinetune":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-probwordnoise-moviereviewsfinetune2")
    elif TRAIN_NOISE_TYPE=="none":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/cnn-lstm-none")
    else:
        raise Exception("invalid TRAIN_NOISE_TYPE")
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    VOCAB_PATH = os.path.join(CHECKPOINT_PATH,"vocab.pkl")

    # settings
    print("#########################"+"\n")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    START_EPOCH, N_EPOCHS = START_EPOCH, 50
    TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = 32, 32

    #############################################
    # load train data (if required)
    #############################################

    TRAIN_TEST_FILE_PATH = os.path.join(BASE_PATH, "traintest/")

    if START_EPOCH>0:

        if FINETUNE!="":
            print("loading vocab for finetuning")
            print(f"loading vocab from {SRC_VOCAB_PATH}")
            vocab = load_vocab_dict(SRC_VOCAB_PATH)
            save_vocab_dict(VOCAB_PATH, vocab)
            # load traintest data
            if TRAIN_NOISE_TYPE=="bea40kfinetune":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.bea40k", "train.bea40k.noise")
                train_data, valid_data = train_validation_split(train_data, 0.90, seed=11690)
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="moviereviewsfinetune":
                #
                train_data_clean = get_lines(os.path.join(TRAIN_TEST_FILE_PATH, "train.moviereviews"))
                train_data_noise1 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25])
                train_data_noise2 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[1.00,0.00,0.00,0.00])
                train_data_noise3 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,1.00,0.00,0.00])
                train_data_noise4 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,1.00,0.00])
                train_data_noise5 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,0.00,1.00])
                train_data_noise = train_data_noise1+train_data_noise2+train_data_noise3+train_data_noise4+train_data_noise5
                train_data_clean = train_data_clean*5
                train_data = [(a,b) for a,b in zip(train_data_clean,train_data_noise)]
                #
                valid_data_clean = get_lines(os.path.join(TRAIN_TEST_FILE_PATH, "valid.moviereviews"))
                valid_data_noise1 = get_line_representation(valid_data_clean,rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25])
                valid_data_noise2 = get_line_representation(valid_data_clean,rep_list=['swap','drop','add','key'], probs=[1.00,0.00,0.00,0.00])
                valid_data_noise3 = get_line_representation(valid_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,1.00,0.00,0.00])
                valid_data_noise4 = get_line_representation(valid_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,1.00,0.00])
                valid_data_noise5 = get_line_representation(valid_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,0.00,1.00])
                valid_data_noise = valid_data_noise1+valid_data_noise2+valid_data_noise3+valid_data_noise4+valid_data_noise5
                valid_data_clean = valid_data_clean*5
                valid_data = [(a,b) for a,b in zip(valid_data_clean,valid_data_noise)]
                print(len(train_data),len(valid_data))
            else:
                raise Exception("invalid TRAIN_NOISE_TYPE in finetuning")
        else:          
            # load traintest data
            if TRAIN_NOISE_TYPE=="word":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm.noise.word")
                train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="prob":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm.noise.prob")
                train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="random":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm.noise.random")
                train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="probword":
                train_data1 = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm.noise.prob")
                train_data1, valid_data1 = train_validation_split(train_data1, 0.8, seed=11690)
                print(len(train_data1),len(valid_data1))
                train_data2 = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm.noise.word")
                train_data2, valid_data2 = train_validation_split(train_data2, 0.8, seed=11690)
                print(len(train_data2),len(valid_data2))
                train_data = train_data1+train_data2
                valid_data = valid_data1+valid_data2
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="probword_v2":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.1blm.v2", "train.1blm.v2.noise.probword")
                train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
                print(len(train_data),len(valid_data))
            elif TRAIN_NOISE_TYPE=="none":
                train_data = load_data(TRAIN_TEST_FILE_PATH, "train.1blm", "train.1blm")
                train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
                print(len(train_data),len(valid_data))        
            else:
                raise Exception("invalid TRAIN_NOISE_TYPE")

    #############################################
    # load vocab
    #############################################

    if START_EPOCH!=1: # if not training from scratch or for inference
        print(f"loading vocab from {VOCAB_PATH}")
        vocab = load_vocab_dict(VOCAB_PATH)
    else:
        # load a vocab for reference
        vocab_ref = {}
        # opfile = open(os.path.join(BASE_PATH, "vocab/phonemedataset.txt"),"r")
        # for line in opfile: vocab_ref.update( {line.strip():0} )
        # opfile.close()

        print(f"loading vocab from train data itself and saving it at {VOCAB_PATH}") 
        vocab = get_tokens([i[0] for i in train_data],
                           keep_simple=True,
                           min_max_freq=(2,float("inf")),
                           topk=100000,
                           intersect=vocab_ref,
                           load_char_tokens=True)
        save_vocab_dict(VOCAB_PATH, vocab)

    if START_EPOCH>0:
        # see how many tokens in labels are going to be UNK
        print ( num_unk_tokens([i[0] for i in train_data], vocab) )
        print ( num_unk_tokens([i[0] for i in valid_data], vocab) )
        print("")
        print([*vocab.keys()])
        #print(vocab["token_freq"])
        #print([(idx,vocab["idx2token"][idx]) for idx in range(100)])

    #############################################
    # load CharCNNWordLSTMModel
    #############################################
    
    model = load_model(vocab)

    #############################################
    # training or inference ??!
    #############################################

    if START_EPOCH>0:

        #############################################
        # training and validation
        #############################################

        # running stats
        max_dev_acc, argmax_dev_acc = -1, -1
        patience = 100

        # Create an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # model to device
        model.to(DEVICE)

        # load parameters if not training from scratch
        if START_EPOCH>1:
            model, optimizer, max_dev_acc, argmax_dev_acc = \
                load_pretrained(model, CHECKPOINT_PATH, optimizer=optimizer)
        else:
            print(f"Training model params from scratch")

        # train and eval
        for epoch_id in range(START_EPOCH,N_EPOCHS+1):
            # check for patience
            if (epoch_id-argmax_dev_acc)>patience:
                print("patience count reached. early stopping initiated")
                print("max_dev_acc: {}, argmax_dev_acc: {}".format(max_dev_acc, argmax_dev_acc))
                break
            # print epoch
            print(f"In epoch: {epoch_id}")
            # train loss and backprop
            train_loss = 0.
            train_acc = 0.
            train_acc_count = 0.
            print("train_data size: {}".format(len(train_data)))
            train_data_iter = batch_iter(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            for batch_id, (batch_labels,batch_sentences) in enumerate(train_data_iter):
                optimizer.zero_grad()
                st_time = time.time()
                # set batch data
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                batch_idxs, batch_lengths_ = char_tokenize(batch_sentences, vocab)
                assert (batch_lengths_==batch_lengths).all()==True
                batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                batch_lengths = batch_lengths.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                # forward
                model.train()
                loss = model(batch_idxs, batch_lengths, targets=batch_labels)
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                loss.backward()
                optimizer.step()
                # compute accuracy in numpy
                if batch_id%10000==0:
                    train_acc_count += 1
                    model.eval()
                    with torch.no_grad():
                        _, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels)
                    model.train()
                    batch_labels = batch_labels.cpu().detach().numpy()
                    batch_lengths = batch_lengths.cpu().detach().numpy()
                    ncorr,ntotal = batch_accuracy_func(batch_predictions,batch_labels,batch_lengths)
                    batch_acc = ncorr/ntotal
                    train_acc += batch_acc     
                # update progress
                progressBar(batch_id+1,
                            int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE)), 
                            ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"],
                            [time.time()-st_time,batch_loss,train_loss/(batch_id+1),batch_acc,train_acc/train_acc_count]) 
            print(f"\nEpoch {epoch_id} train_loss: {train_loss/(batch_id+1)}")

            # valid loss
            valid_loss = 0.
            valid_acc = 0.
            print("valid_data size: {}".format(len(valid_data)))
            valid_data_iter = batch_iter(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
            for batch_id, (batch_labels,batch_sentences) in enumerate(valid_data_iter):
                st_time = time.time()
                # set batch data
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                batch_idxs, batch_lengths_ = char_tokenize(batch_sentences, vocab)
                assert (batch_lengths_==batch_lengths).all()==True
                batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                batch_lengths = batch_lengths.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                # forward
                model.eval()
                with torch.no_grad():
                    batch_loss, batch_predictions = model(batch_idxs, batch_lengths, targets=batch_labels)
                model.train()        
                valid_loss += batch_loss
                # compute accuracy in numpy
                batch_labels = batch_labels.cpu().detach().numpy()
                batch_lengths = batch_lengths.cpu().detach().numpy()
                ncorr,ntotal = batch_accuracy_func(batch_predictions,batch_labels,batch_lengths)
                batch_acc = ncorr/ntotal
                valid_acc += batch_acc
                # update progress
                progressBar(batch_id+1,
                            int(np.ceil(len(valid_data) / VALID_BATCH_SIZE)), 
                            ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                            [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),batch_acc,valid_acc/(batch_id+1)]) 
            print(f"\nEpoch {epoch_id} valid_loss: {valid_loss/(batch_id+1)}")

            # save model, optimizer and test_predictions if val_acc is improved
            if valid_acc>=max_dev_acc:

                # to file
                name = "model.pth.tar" # "model-epoch{}.pth.tar".format(epoch_id)
                torch.save({
                    'epoch_id': epoch_id,
                    'max_dev_acc': max_dev_acc,
                    'argmax_dev_acc': argmax_dev_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(CHECKPOINT_PATH,name))
                print("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH,name),epoch_id))

                # re-assign
                max_dev_acc, argmax_dev_acc = valid_acc, epoch_id

    else:

        #############################################
        # inference
        #############################################

        # load parameters
        model = load_pretrained(model, CHECKPOINT_PATH)

        # infer
        TRAIN_TEST_FILE_PATH1 = os.path.join(BASE_PATH, "traintest")
        TRAIN_TEST_FILE_PATH2 = os.path.join(BASE_PATH, "traintest/wo_context")
        '''
        paths = [TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2]
        files1 = ["test.bea60k","test.1blm","test.1blm","combined_data","aspell_big","aspell_small"]
        files2 = ["test.bea60k.noise","test.1blm.noise.prob","test.1blm.noise.word","combined_data.noise","aspell_big.noise","aspell_small.noise"]
        INFER_BATCH_SIZE = 16
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2]
        files1 = ["combined_data","aspell_big","aspell_small"]
        files2 = ["combined_data.noise","aspell_big.noise","aspell_small.noise"]
        INFER_BATCH_SIZE = 1024
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.1blm"]
        files2 = ["test.1blm.noise.prob"]
        INFER_BATCH_SIZE = 24 #128
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.1blm"]
        files2 = ["test.1blm.noise.word"]
        INFER_BATCH_SIZE = 24 # 128
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.1blm"]
        files2 = ["test.1blm.noise.random"]
        INFER_BATCH_SIZE = 20 # 128
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea4k"]
        files2 = ["test.bea4k.noise"]
        INFER_BATCH_SIZE = 8
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea60k"]
        files2 = ["test.bea60k.noise"]
        INFER_BATCH_SIZE = 8
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea60k.ambiguous_artificial"]
        files2 = ["test.bea60k.ambiguous_artificial.noise"]
        INFER_BATCH_SIZE = 4
        '''
        # '''
        paths = [TRAIN_TEST_FILE_PATH1, TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea60k.ambiguous_natural_v7", "test.bea60k.ambiguous_natural_v8"]
        files2 = ["test.bea60k.ambiguous_natural_v7.noise", "test.bea60k.ambiguous_natural_v8.noise"]
        INFER_BATCH_SIZE = 8
        # '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.jfleg"]
        files2 = ["test.jfleg.noise"]
        INFER_BATCH_SIZE = 8
        '''
        for x,y,z in zip(paths,files1,files2):
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE)

