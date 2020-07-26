
#############################################
# USAGE
# CUDA_VISIBLE_DEVICES=0 python sclstmelmo.py probword ../../data -1
#
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py none ../../data 1
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py random ../../data 1
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py word ../../data 1
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py prob ../../data 1
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py probword ../../data 1
# CUDA_VISIBLE_DEVICES=1 python sclstmelmo.py probword_v2 ../../data 1
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
import numpy as np
import re
import time
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
# print(torch.cuda.current_device())
# torch.cuda.set_device(1)
# print(torch.cuda.current_device())
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else "cpu"
DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"

from helpers import progressBar
from helpers import load_vocab_dict, save_vocab_dict
from helpers import load_data, train_validation_split, get_char_tokens, get_tokens, num_unk_tokens
from helpers import batch_iter, labelize, tokenize, char_tokenize, sclstm_tokenize
from helpers import untokenize, untokenize_without_unks, untokenize_without_unks2, untokenize_without_unks3, get_model_nparams
from helpers import batch_accuracy_func

from helpers2 import get_line_representation, get_lines

from models import ElmoSCLSTM
from allennlp.modules.elmo import batch_to_ids as elmo_batch_to_ids

from evals import get_metrics

"""
NEW: reranking snippets
"""
# (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
import torch
from torch.nn import CrossEntropyLoss
HFACE_BATCH_SIZE = 8

# from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
# gpt2Tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
# gpt2LMHeadModel = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
# gpt2Tokenizer.add_special_tokens({'pad_token':"[PAD]"})
# gpt2LMHeadModel.resize_token_embeddings(len(gpt2Tokenizer))
# assert gpt2Tokenizer.pad_token == '[PAD]'

from transformers import GPT2Tokenizer, GPT2LMHeadModel
gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2-medium')
gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token

# from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
# txlTokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# txlLMHeadModel = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
# txlTokenizer.pad_token = txlTokenizer.eos_token



def get_losses_from_gpt_lm(this_sents: "list[str]", gpt2LMHeadModel, gpt2Tokenizer, DEVICE):

    this_input_ids = gpt2Tokenizer.batch_encode_plus(this_sents, add_special_tokens=True, pad_to_max_length=True, add_space_before_punct_symbol=True)["input_ids"]
    this_labels = torch.tensor([[i if i!=gpt2Tokenizer.pad_token_id else -100 for i in row] for row in this_input_ids]).to(DEVICE)
    this_input_ids = torch.tensor(this_input_ids).to(DEVICE)
    this_outputs = gpt2LMHeadModel(input_ids=this_input_ids)
    this_lm_logits = this_outputs[0]
    # Shift so that tokens < n predict n
    shift_logits2 = this_lm_logits[:, :-1, :]
    shift_labels2 = this_labels[:, 1:]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits2.permute(0,2,1), shift_labels2)
    losses = loss.sum(dim=-1).cpu().detach().numpy().tolist()

    return losses

def get_losses_from_txl_lm(this_sents: "list[str]", txlLMHeadModel, txlTokenizer, DEVICE):

    this_input_ids_dict = txlTokenizer.batch_encode_plus(this_sents, add_special_tokens=True, pad_to_max_length=True, add_space_before_punct_symbol=True)
    this_input_ids = this_input_ids_dict["input_ids"]
    chunks = [sum(val) for val in this_input_ids_dict["attention_mask"]]
    chunks_cumsum = np.cumsum(chunks).tolist()

    this_labels = torch.tensor([[i if i!=txlTokenizer.pad_token_id else -100 for i in row] for row in this_input_ids]).to(DEVICE)
    this_input_ids = torch.tensor(this_input_ids).to(DEVICE)
    this_outputs = txlLMHeadModel(input_ids=this_input_ids,labels=this_labels)
    this_loss = this_outputs[0]
    this_loss = this_loss.view(-1).cpu().detach().numpy()
    losses = [sum(this_loss[str_pos:end_pos-1]) for str_pos,end_pos in zip([0]+chunks_cumsum[:-1],chunks_cumsum)]

    return losses

def load_model(vocab, verbose=False):

    model = ElmoSCLSTM(3*len(vocab["chartoken2idx"]),vocab["token2idx"][ vocab["pad_token"] ],len(vocab["token_freq"]),early_concat=False)
    if verbose:
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
    print(f"previously, max_dev_acc: {max_dev_acc:.5f} and argmax_dev_acc: {argmax_dev_acc:.5f}")

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc
    
    return model

def model_predictions(model, data, vocab, DEVICE, BATCH_SIZE=16, backoff="pass-through"):
    """
    model: an instance of ElmoSCLSTM
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    """
    
    topk = 1
    # print("###############################################")
    # inference_st_time = time.time()
    final_sentences = []
    VALID_BATCH_SIZE = BATCH_SIZE
    # print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    for batch_id, (batch_clean_sentences,batch_corrupt_sentences) in enumerate(data_iter):
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = sclstm_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_==batch_lengths).all()==True
        batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_corrupt_sentences]).to(DEVICE)
        # forward
        with torch.no_grad():
            """
            NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
            """
            _, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels, topk=topk)
        batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff=backoff)
        final_sentences.extend(batch_predictions)
    # print("total inference time for this data is: {:4f} secs".format(time.time()-inference_st_time))
    return final_sentences

def model_inference(model, data, topk, DEVICE, BATCH_SIZE=16, beam_search=False, selected_lines_file=None, vocab_=None):
    """
    model: an instance of ElmoSCLSTM
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    topk: how many of the topk softmax predictions are considered for metrics calculations
    DEVICE: "cuda:0" or "cpu"
    BATCH_SIZE: batch size for input to the model
    beam_search: if True, greedy topk will not be performed
    """
    if vocab_ is not None:
        vocab = vocab_
    if beam_search:
        if topk<2:
            raise Exception("when using beam_search, topk must be greater than 1, topk is used as beam width")
        else:
            print(f":: doing BEAM SEARCH with topk:{topk} ::")

        if selected_lines_file is not None:
             raise Exception("when using beam_search, ***selected_lines_file*** arg is not used; no implementation")

    # list of dicts with keys {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
    results = []
    line_index = 0 

    inference_st_time = time.time()
    VALID_BATCH_SIZE = BATCH_SIZE
    valid_loss, valid_acc = 0., 0.
    corr2corr, corr2incorr, incorr2corr, incorr2incorr = 0, 0, 0, 0
    predictions = []
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    for batch_id, (batch_clean_sentences,batch_corrupt_sentences) in tqdm(enumerate(data_iter)):
        torch.cuda.empty_cache()
        # st_time = time.time()
        # set batch data
        batch_labels, batch_lengths = labelize(batch_clean_sentences, vocab)
        batch_idxs, batch_lengths_ = sclstm_tokenize(batch_corrupt_sentences, vocab)
        assert (batch_lengths_==batch_lengths).all()==True
        batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_corrupt_sentences]).to(DEVICE)
        # forward
        try:
            with torch.no_grad():
                if not beam_search:
                    """
                    NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len) if topk==1
                    """
                    batch_loss, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels, topk=topk) # topk=1 or 5
                else:
                    """
                    NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk==None
                    """
                    batch_loss, batch_predictions, batch_predictions_probs = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels, topk=topk, beam_search=True)
        except RuntimeError:
            print(f"batch_idxs:{len(batch_idxs)},batch_lengths:{batch_lengths.shape},batch_elmo_inp:{batch_elmo_inp.shape},batch_labels:{batch_labels.shape}")
            raise Exception("")
        valid_loss += batch_loss
        # compute accuracy in numpy
        batch_labels = batch_labels.cpu().detach().numpy()
        batch_lengths = batch_lengths.cpu().detach().numpy()
        # based on beam_search, do either greedy topk or beam search for topk
        if not beam_search:
            # based on topk, obtain either strings of batch_predictions or list of tokens
            if topk==1:
                batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_corrupt_sentences)    
            else:
                batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_corrupt_sentences)
            predictions.extend(batch_predictions)
            
            # batch_clean_sentences = [line.lower() for line in batch_clean_sentences]
            # batch_corrupt_sentences = [line.lower() for line in batch_corrupt_sentences]
            # batch_predictions = [line.lower() for line in batch_predictions]
            corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
                get_metrics(batch_clean_sentences,batch_corrupt_sentences,batch_predictions,check_until_topk=topk,return_mistakes=False)
            corr2corr+=corr2corr_
            corr2incorr+=corr2incorr_
            incorr2corr+=incorr2corr_
            incorr2incorr+=incorr2incorr_

            for i, (a,b,c) in enumerate(zip(batch_clean_sentences,batch_corrupt_sentences,batch_predictions)):
                results.append({"id":line_index+i, "original":a, "noised":b, "predicted":c, "topk":[], "topk_prediction_probs":[], "topk_reranker_losses":[]})
            line_index += len(batch_clean_sentences)

        else:
            """
            NEW: use untokenize_without_unks3 for beam search outputs
            """
            # k different lists each of type batch_predictions as in topk==1
            # List[List[Strings]]
            k_batch_predictions, k_batch_predictions_probs = untokenize_without_unks3(batch_predictions, batch_predictions_probs, batch_lengths, vocab, batch_clean_sentences, topk)

            ##########################################################
            ############## this takes top1 as-is #####################
            # corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
            #     get_metrics(batch_clean_sentences,batch_corrupt_sentences,k_batch_predictions[0],check_until_topk=1,return_mistakes=False)
            # corr2corr+=corr2corr_
            # corr2incorr+=corr2incorr_
            # incorr2corr+=incorr2corr_
            # incorr2incorr+=incorr2incorr_

            ##########################################################
            ############### this does reranking ######################
            gpt2LMHeadModel.to(DEVICE)
            gpt2LMHeadModel.eval()
            # txlLMHeadModel.to(DEVICE)
            # txlLMHeadModel.eval()

            reranked_batch_predictions = []
            batch_clean_sentences_ = []
            batch_corrupt_sentences_ = []
            batch_losses_ = []
            with torch.no_grad():    
                for b in range(len(batch_clean_sentences)):
                    losses = []
                    this_sents = [k_batch_predictions[k][b] for k in range(topk)]
                    losses = get_losses_from_gpt_lm(this_sents, gpt2LMHeadModel, gpt2Tokenizer, DEVICE)
                    # losses = get_losses_from_txl_lm(this_sents, txlLMHeadModel, txlTokenizer, DEVICE)
                    kmin = np.argmin(losses)
                    reranked_batch_predictions.append(k_batch_predictions[kmin][b])
                    batch_clean_sentences_.append(batch_clean_sentences[b])
                    batch_corrupt_sentences_.append(batch_corrupt_sentences[b])
                    batch_losses_.append(losses)

            corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
                get_metrics(batch_clean_sentences_,batch_corrupt_sentences_,reranked_batch_predictions,check_until_topk=1,return_mistakes=False)
            corr2corr+=corr2corr_
            corr2incorr+=corr2incorr_
            incorr2corr+=incorr2corr_
            incorr2incorr+=incorr2incorr_

            batch_predictions_k = [[k_batch_predictions[j][i] for j in range(len(k_batch_predictions))] for i in range(len(k_batch_predictions[0]))]
            batch_predictions_probs_k = [[k_batch_predictions_probs[j][i] for j in range(len(k_batch_predictions_probs))] for i in range(len(k_batch_predictions_probs[0]))]
            for i, (a,b,c,d,e,f) in \
                enumerate(zip(batch_clean_sentences_,batch_corrupt_sentences_,reranked_batch_predictions,batch_predictions_k,batch_predictions_probs_k,batch_losses_)):
                results.append({"id":line_index+i, "original":a, "noised":b, "predicted":c, "topk":d, "topk_prediction_probs":e, "topk_reranker_losses":f})
            line_index += len(batch_clean_sentences)

        # delete
        del batch_loss
        del batch_predictions
        del batch_labels, batch_lengths, batch_idxs, batch_lengths_, batch_elmo_inp
        torch.cuda.empty_cache()

        # '''
        # # update progress
        # progressBar(batch_id+1,
        #             int(np.ceil(len(data) / VALID_BATCH_SIZE)), 
        #             ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
        #             [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        # '''

    print(f"\nEpoch {None} valid_loss: {valid_loss/(batch_id+1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time()-inference_st_time))
    print("###############################################")
    print("total token count: {}".format(corr2corr+corr2incorr+incorr2corr+incorr2incorr))
    print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
    print(f"accuracy is {(corr2corr+incorr2corr)/(corr2corr+corr2incorr+incorr2corr+incorr2incorr)}")
    print(f"word correction rate is {(incorr2corr)/(incorr2corr+incorr2incorr)}")
    print("###############################################")

    if not beam_search and selected_lines_file is not None:

        print("evaluating only for selected lines ... ")

        assert len(data)==len(predictions), print(len(data),len(predictions),"lengths mismatch")

        if selected_lines_file is not None:
            selected_lines = {num:"" for num in [int(line.strip()) for line in open(selected_lines_file,'r')]}
        else:
            selected_lines = None

        clean_lines, corrupt_lines,predictions_lines = [tpl[0] for tpl in data], [tpl[1] for tpl in data], predictions

        corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes  = \
            get_metrics(clean_lines,corrupt_lines,predictions_lines,return_mistakes=True,selected_lines=selected_lines)

        print("###############################################")
        print("total token count: {}".format(corr2corr+corr2incorr+incorr2corr+incorr2incorr))
        print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")
        print(f"accuracy is {(corr2corr+incorr2corr)/(corr2corr+corr2incorr+incorr2corr+incorr2incorr)}")
        print(f"word correction rate is {(incorr2corr)/(incorr2corr+incorr2incorr)}")
        print("###############################################")
    
    return results



if __name__=="__main__":

    print("#########################"+"\n")
    # "word", "prob", "probword", 'random', bea40kfinetune', 'moviereviewsfinetune'
    TRAIN_NOISE_TYPE = sys.argv[1]
    # "../../data"
    BASE_PATH = sys.argv[2]
    # -ve value for inference only; 1 for training a new model from scratch; >1 for continuing training
    START_EPOCH = int(sys.argv[3])
    if START_EPOCH==0:
        raise Exception("START_EPOCH must be a non-zero value; If starting from scratch, use 1 instead of 0")
    # :NEW: finetune now from a specific epoch of a model
    # "probword"
    if len(sys.argv)>4:
        FINETUNE = sys.argv[4]
        if FINETUNE=='probword':
            SRC_CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-probwordnoise")
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
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-wordnoise")
    elif TRAIN_NOISE_TYPE=="prob":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-probnoise")
    elif TRAIN_NOISE_TYPE=="random":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-randomnoise")
    elif TRAIN_NOISE_TYPE=="probword":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-probwordnoise")
    elif TRAIN_NOISE_TYPE=="bea40kfinetune":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-probwordnoise-bea40kfinetune")
    elif TRAIN_NOISE_TYPE=="moviereviewsfinetune":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-probwordnoise-moviereviewsfinetune2")
    elif TRAIN_NOISE_TYPE=="none":
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/scrnnelmo-none")
    else:
        raise Exception("invalid TRAIN_NOISE_TYPE")
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    VOCAB_PATH = os.path.join(CHECKPOINT_PATH,"vocab.pkl")

    # settings
    print("#########################"+"\n")
    START_EPOCH, N_EPOCHS = START_EPOCH, 50
    TRAIN_BATCH_SIZE, VALID_BATCH_SIZE =  32, 32 # 16, 16

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
    # load ElmoSCLSTM
    #############################################

    model = load_model(vocab, verbose=False)

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
            # file to write progress to
            progress_write_file = open(os.path.join(CHECKPOINT_PATH,f"progress_retrain_from_epoch{START_EPOCH}.txt"),'w')
            # model and optimizer load_state_dict
            if FINETUNE!="":
                print("loading pretrained weights for finetuning")
                print(f"loading pretrained weights from {SRC_CHECKPOINT_PATH}")
                model, optimizer, _, _ = load_pretrained(model, SRC_CHECKPOINT_PATH, optimizer=optimizer)
                progress_write_file.write(f"Training model params after loading from path: {SRC_CHECKPOINT_PATH}\n") 
            else:
                print(f"loading pretrained weights from {CHECKPOINT_PATH}")
                model, optimizer, max_dev_acc, argmax_dev_acc = load_pretrained(model, CHECKPOINT_PATH, optimizer=optimizer)
                progress_write_file.write(f"Training model params after loading from path: {CHECKPOINT_PATH}\n") 
        else:
            # file to write progress to
            progress_write_file = open(os.path.join(CHECKPOINT_PATH,"progress.txt"),'w')
            print(f"Training model params from scratch")
            progress_write_file.write(f"Training model params from scratch\n")
        progress_write_file.flush()

        # train and eval
        for epoch_id in range(START_EPOCH,N_EPOCHS+1):
            # check for patience
            if (epoch_id-argmax_dev_acc)>patience:
                print("patience count reached. early stopping initiated")
                print("max_dev_acc: {}, argmax_dev_acc: {}".format(max_dev_acc, argmax_dev_acc))
                break
            # if finetuning and the noise type is moviereviews,
            #    create a different train data every epoch
            if TRAIN_NOISE_TYPE=="moviereviewsfinetune":
                train_data_clean = get_lines(os.path.join(TRAIN_TEST_FILE_PATH, "train.moviereviews"))
                train_data_noise1 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25])
                train_data_noise2 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[1.00,0.00,0.00,0.00])
                train_data_noise3 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,1.00,0.00,0.00])
                train_data_noise4 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,1.00,0.00])
                train_data_noise5 = get_line_representation(train_data_clean,rep_list=['swap','drop','add','key'], probs=[0.00,0.00,0.00,1.00])
                train_data_noise = train_data_noise1+train_data_noise2+train_data_noise3+train_data_noise4+train_data_noise5
                train_data_clean = train_data_clean*5
                train_data = [(a,b) for a,b in zip(train_data_clean,train_data_noise)]
                print(f"new training instances created, train data size now: {len(train_data)}")
            # print epoch
            print(f"In epoch: {epoch_id}")
            progress_write_file.write(f"In epoch: {epoch_id}\n")
            progress_write_file.flush()
            # train loss and backprop
            train_loss = 0.
            train_acc = 0.
            train_acc_count = 0.
            print("train_data size: {}".format(len(train_data)))
            progress_write_file.write("train_data size: {}\n".format(len(train_data)))
            progress_write_file.flush()
            train_data_iter = batch_iter(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            #for batch_id, (batch_labels,batch_sentences) in tqdm(enumerate(train_data_iter)):
            for batch_id, (batch_labels,batch_sentences) in enumerate(train_data_iter):
                optimizer.zero_grad()
                st_time = time.time()
                # set batch data
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                batch_idxs, batch_lengths_ = sclstm_tokenize(batch_sentences, vocab)
                assert (batch_lengths_==batch_lengths).all()==True
                batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                batch_lengths = batch_lengths.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_sentences]).to(DEVICE)
                # forward
                model.train()
                loss = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels)
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                loss.backward()
                optimizer.step()
                # compute accuracy in numpy
                if batch_id%1000==0:
                    train_acc_count += 1
                    model.eval()
                    with torch.no_grad():
                        _, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels)
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
                if batch_id==0 or (batch_id+1)%5000==0:
                    nb = int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE))
                    progress_write_file.write(f"{batch_id+1}/{nb}\n")
                    progress_write_file.write(f"batch_time: {time.time()-st_time}, avg_batch_loss: {train_loss/(batch_id+1)}, avg_batch_acc: {train_acc/train_acc_count}\n")
                    progress_write_file.flush()
            print(f"\nEpoch {epoch_id} train_loss: {train_loss/(batch_id+1)}")

            try:
                # valid loss
                valid_loss = 0.
                valid_acc = 0.
                print("valid_data size: {}".format(len(valid_data)))
                progress_write_file.write("valid_data size: {}\n".format(len(valid_data)))
                progress_write_file.flush()
                valid_data_iter = batch_iter(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
                for batch_id, (batch_labels,batch_sentences) in enumerate(valid_data_iter):
                    st_time = time.time()
                    # set batch data
                    batch_labels, batch_lengths = labelize(batch_labels, vocab)
                    batch_idxs, batch_lengths_ = sclstm_tokenize(batch_sentences, vocab)
                    assert (batch_lengths_==batch_lengths).all()==True
                    batch_idxs = [batch_idxs_.to(DEVICE) for batch_idxs_ in batch_idxs]
                    batch_lengths = batch_lengths.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    batch_elmo_inp = elmo_batch_to_ids([line.split() for line in batch_sentences]).to(DEVICE)
                    # forward
                    model.eval()
                    with torch.no_grad():
                        batch_loss, batch_predictions = model(batch_idxs, batch_lengths, batch_elmo_inp, targets=batch_labels)
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
                    if batch_id==0 or (batch_id+1)%2000==0:
                        nb = int(np.ceil(len(valid_data) / VALID_BATCH_SIZE))
                        progress_write_file.write(f"{batch_id}/{nb}\n")
                        progress_write_file.write(f"batch_time: {time.time()-st_time}, avg_batch_loss: {valid_loss/(batch_id+1)}, avg_batch_acc: {valid_acc/(batch_id+1)}\n")
                        progress_write_file.flush()
                print(f"\nEpoch {epoch_id} valid_loss: {valid_loss/(batch_id+1)}")

                # save model, optimizer and test_predictions if val_acc is improved
                if valid_acc>=max_dev_acc:

                    # to file
                    #name = "model-epoch{}.pth.tar".format(epoch_id)
                    name = "model.pth.tar".format(epoch_id)
                    torch.save({
                        'epoch_id': epoch_id,
                        'previous_max_dev_acc': max_dev_acc,
                        'previous_argmax_dev_acc': argmax_dev_acc,
                        'max_dev_acc': valid_acc,
                        'argmax_dev_acc': epoch_id,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        os.path.join(CHECKPOINT_PATH,name))
                    print("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH,name),epoch_id))

                    # re-assign
                    max_dev_acc, argmax_dev_acc = valid_acc, epoch_id

            except Exception as e:
                    temp_folder = os.path.join(CHECKPOINT_PATH,"temp")
                    if not os.path.exists(temp_folder):
                        os.makedirs(temp_folder)
                    name = "model.pth.tar".format(epoch_id)
                    torch.save({
                        'epoch_id': epoch_id,
                        'previous_max_dev_acc': max_dev_acc,
                        'previous_argmax_dev_acc': argmax_dev_acc,
                        'max_dev_acc': valid_acc,
                        'argmax_dev_acc': epoch_id,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        os.path.join(temp_folder,name))
                    print("Model saved at {} in epoch {}".format(os.path.join(temp_folder,name),epoch_id))
                    raise Exception(e)
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
        paths = [TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2]
        files1 = ["combined_data","aspell_big","aspell_small"]
        files2 = ["combined_data.noise","aspell_big.noise","aspell_small.noise"]
        INFER_BATCH_SIZE = 1024

        for x,y,z in zip(paths,files1,files2):
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE)
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1]
        files1 = ["test.jfleg","test.bea4k","test.bea60k"]
        files2 = ["test.jfleg.noise","test.bea4k.noise","test.bea60k.noise"]
        INFER_BATCH_SIZE = 8
        selected_lines_file = None

        # expect a dict as {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
        for x,y,z in zip(paths,files1,files2):
            print("\n\n\n\n")
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            greedy_results = model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=False,selected_lines_file=selected_lines_file)
            # beam_search_results = model_inference(model,test_data,topk=10,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=True)
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1, TRAIN_TEST_FILE_PATH1, TRAIN_TEST_FILE_PATH1]
        files1 = ["test.1blm", "test.1blm", "test.1blm"]
        files2 = ["test.1blm.noise.random", "test.1blm.noise.prob", "test.1blm.noise.word"]
        INFER_BATCH_SIZE = 20
        selected_lines_file = None

        # expect a dict as {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
        for x,y,z in zip(paths,files1,files2):
            print("\n\n\n\n")
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            greedy_results = model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=False,selected_lines_file=selected_lines_file)
            # beam_search_results = model_inference(model,test_data,topk=10,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=True)
        '''
        '''
        paths = [TRAIN_TEST_FILE_PATH1, TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea60k.ambiguous_natural_v7", "test.bea60k.ambiguous_natural_v8"]
        files2 = ["test.bea60k.ambiguous_natural_v7.noise", "test.bea60k.ambiguous_natural_v8.noise"]
        INFER_BATCH_SIZE = 8
        selected_lines_file = None

        # expect a dict as {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
        for x,y,z in zip(paths,files1,files2):
            print("\n\n\n\n")
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            greedy_results = model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=False,selected_lines_file=selected_lines_file)
            # beam_search_results = model_inference(model,test_data,topk=10,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=True)
        '''
        # '''
        paths = [TRAIN_TEST_FILE_PATH1]
        files1 = ["test.bea60k"]
        files2 = ["test.bea60k.noise"]
        INFER_BATCH_SIZE = 8
        selected_lines_file = None

        # expect a dict as {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
        for x,y,z in zip(paths,files1,files2):
            print("\n\n\n\n")
            print(x,y,z)
            test_data = load_data(x,y,z)
            print ( num_unk_tokens([i[0] for i in test_data], vocab) )
            greedy_results = model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=False,selected_lines_file=selected_lines_file)
            # beam_search_results = model_inference(model,test_data,topk=10,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=True)
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH1,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2]
        # files1 = ["test.bea60k","test.1blm","test.1blm","combined_data","aspell_big","aspell_small"]
        # files2 = ["test.bea60k.noise","test.1blm.noise.prob","test.1blm.noise.word","combined_data.noise","aspell_big.noise","aspell_small.noise"]
        # INFER_BATCH_SIZE = 16
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2,TRAIN_TEST_FILE_PATH2]
        # files1 = ["combined_data","aspell_big","aspell_small"]
        # files2 = ["combined_data.noise","aspell_big.noise","aspell_small.noise"]
        # INFER_BATCH_SIZE = 1024
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.1blm","test.1blm"]
        # files2 = ["test.1blm.noise.prob","test.1blm.noise.word"]
        # INFER_BATCH_SIZE = 64 # 128
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.1blm"]
        # files2 = ["test.1blm.noise.prob"]
        # INFER_BATCH_SIZE = 32 #64 #128
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_probnoise"
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.1blm"]
        # files2 = ["test.1blm.noise.word"]
        # INFER_BATCH_SIZE = 32 #64 #128
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_wordnoise"
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1, TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.1blm","test.1blm"]
        # files2 = ["test.1blm.noise.prob","test.1blm.noise.word"]
        # INFER_BATCH_SIZE = 32
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.bea4k",]
        # files2 = ["test.bea4k.noise"]
        # INFER_BATCH_SIZE = 16
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_bea4k"
        # selected_lines_file = None # "../gec-pseudodata/test.bea4k.lines.txt" # None
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.bea60k"]
        # files2 = ["test.bea60k.noise"]
        # INFER_BATCH_SIZE = 10
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_bea60k"
        # selected_lines_file = None # "../gec-pseudodata/test.bea60k.lines.txt" # None
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.bea20k"]
        # files2 = ["test.bea20k.noise"]
        # INFER_BATCH_SIZE = 10
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_bea20k"
        # selected_lines_file = None # "../gec-pseudodata/test.bea20k.lines.txt" # None
        # '''
        # '''
        # paths = [TRAIN_TEST_FILE_PATH1]
        # files1 = ["test.jfleg"]
        # files2 = ["test.jfleg.noise"]
        # INFER_BATCH_SIZE = 32
        # ANALYSIS_DIR = f"./analysis_{TRAIN_NOISE_TYPE}_jfleg"
        # selected_lines_file = None # "../gec-pseudodata/test.jfleg.lines.txt" # None
        # '''
        
        # # expect a dict as {"id":, "original":, "noised":, "predicted":, "topk":, "topk_prediction_probs":, "topk_reranker_losses":,}
        # for x,y,z in zip(paths,files1,files2):
        #     print("\n\n\n\n")
        #     print(x,y,z)
        #     test_data = load_data(x,y,z)
        #     print ( num_unk_tokens([i[0] for i in test_data], vocab) )
        #     greedy_results = model_inference(model,test_data,topk=1,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=False,selected_lines_file=selected_lines_file)
        #     # beam_search_results = model_inference(model,test_data,topk=10,DEVICE=DEVICE,BATCH_SIZE=INFER_BATCH_SIZE,beam_search=True)

        # ANALYSIS_DIR = os.path.join("scrnnelmo",ANALYSIS_DIR)
        # if not os.path.exists(ANALYSIS_DIR):
        #     os.makedirs(ANALYSIS_DIR)
        # import jsonlines
        # #
        # print("greedy...")
        # greedy_lines_fully_correct = {line["id"]:"" for line in greedy_results if line["original"]==line["predicted"]}
        # greedy_lines_otherwise = {line["id"]:"" for line in greedy_results if line["original"]!=line["predicted"]}
        # print(f'# Lines Predicted Fully Correct: {len(greedy_lines_fully_correct)}')
        # print(f'# Lines Otherwise: {len(greedy_lines_otherwise)}')
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"greedy_results.jsonl"),'w')
        # for line in greedy_results: opfile.write(line)
        # opfile.close()
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"greedy_results_corr_preds.jsonl"),'w')
        # for line in [line for line in greedy_results if line["original"]==line["predicted"]]: opfile.write(line)
        # opfile.close()
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"greedy_results_incorr_preds.jsonl"),'w')
        # for line in [line for line in greedy_results if line["original"]!=line["predicted"]]: opfile.write(line)
        # opfile.close()
        #
        # print("beam_search...")
        # beam_search_lines_fully_correct = {line["id"]:"" for line in beam_search_results if line["original"]==line["predicted"]}
        # beam_search_lines_otherwise = {line["id"]:"" for line in beam_search_results if line["original"]!=line["predicted"]}
        # print(f'# Lines Predicted Fully Correct: {len(beam_search_lines_fully_correct)}')
        # print(f'# Lines Otherwise: {len(beam_search_lines_otherwise)}')
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"beam_search_results.jsonl"),'w')
        # for line in beam_search_results: opfile.write(line)
        # opfile.close()
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"beam_search_results_corr_preds.jsonl"),'w')
        # for line in [line for line in beam_search_results if line["original"]==line["predicted"]]: opfile.write(line)
        # opfile.close()
        # opfile = jsonlines.open(os.path.join(ANALYSIS_DIR,"beam_search_results_incorr_preds.jsonl"),'w')
        # for line in [line for line in beam_search_results if line["original"]!=line["predicted"]]: opfile.write(line)
        # opfile.close()
        # #
        # # confusion matrix
        # corr2corr = len([k for k in greedy_lines_fully_correct if k in beam_search_lines_fully_correct])
        # corr2incorr = len([k for k in greedy_lines_fully_correct if k in beam_search_lines_otherwise])
        # incorr2corr = len([k for k in greedy_lines_otherwise if k in beam_search_lines_fully_correct])
        # incorr2incorr = len([k for k in greedy_lines_otherwise if k in beam_search_lines_otherwise])
        # print("Confusion Matrix for before and after beam search: ")
        # print(f"corr2corr:{corr2corr}, corr2incorr:{corr2incorr}, incorr2corr:{incorr2corr}, incorr2incorr:{incorr2incorr}")







#########################################
# reranking snippets from past
#########################################

# if save_dir is not None:
#     line_index = 0
#     analysis_path = save_dir
#     if not os.path.exists(analysis_path): 
#         os.makedirs(analysis_path)
#     if beam_search:
#         line_index_wrong_opfile = open(f"./{analysis_path}/beam_search_wrong.txt","w")
#         line_index_right_opfile = open(f"./{analysis_path}/beam_search_right.txt","w")
#         k_wrong_opfile = open(f"./{analysis_path}/beam_search_k_wrong.txt","w")
#         k_right_opfile = open(f"./{analysis_path}/beam_search_k_right.txt","w")
#     else:
#         line_index_wrong_opfile = open(f"./{analysis_path}/greedy_wrong.txt","w")
#         line_index_right_opfile = open(f"./{analysis_path}/greedy_right.txt","w") 

# reranked_batch_predictions = []
# batch_clean_sentences_ = []
# batch_corrupt_sentences_ = []
# with torch.no_grad():    
#     for b in range(len(batch_clean_sentences)):
#         try:
#             losses = []
#             for sent in [k_batch_predictions[k][b] for k in range(topk)]:
#                 if sent!="" or sent is not None:
#                     input_ids = torch.tensor(gpt2Tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#                     input_ids = input_ids.to(DEVICE)
#                     outputs = gpt2LMHeadModel(input_ids, labels=input_ids)
#                     loss = outputs[0].item()
#                 else:
#                     loss = 10000.0
#                 losses.append(loss)
#             kmin = np.argmin(losses)
#             reranked_batch_predictions.append(k_batch_predictions[kmin][b])
#             batch_clean_sentences_.append(batch_clean_sentences[b])
#             batch_corrupt_sentences_.append(batch_corrupt_sentences[b])
#         except Exception as e:
#             reranked_batch_predictions.append(k_batch_predictions[0][b])
#             batch_clean_sentences_.append(batch_clean_sentences[b])
#             batch_corrupt_sentences_.append(batch_corrupt_sentences[b])
# corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
#     get_metrics(batch_clean_sentences_,batch_corrupt_sentences_,reranked_batch_predictions,check_until_topk=1,return_mistakes=False)
# corr2corr+=corr2corr_
# corr2incorr+=corr2incorr_
# incorr2corr+=incorr2corr_
# incorr2incorr+=incorr2incorr_

# this_batch = [[k_batch_predictions[k][i] for k in range(len(k_batch_predictions))] for i in range(len(k_batch_predictions[0]))]
# flat_batch = sum(this_batch,[]); # print(flat_batch); print(len(flat_batch))
# lens = [len(s) for s in this_batch]
# ii = 0
# flat_losses = []
# model.eval()
# model.to(DEVICE)
# with torch.no_grad():
#     while ii<len(flat_batch):
#         try:
#             curr_batch = flat_batch[ii:ii+HFACE_BATCH_SIZE]
#             curr_inputs = gpt2Tokenizer.batch_encode_plus(curr_batch,pad_to_max_length=True)
#             curr_inputs_ids = curr_inputs["input_ids"]
#             curr_inputs = {k:torch.tensor(v).to(DEVICE) for k,v in curr_inputs.items()}
#             curr_outputs = gpt2LMHeadModel(input_ids=curr_inputs["input_ids"],token_type_ids=curr_inputs["token_type_ids"],attention_mask=curr_inputs["attention_mask"])
#             lm_logits = curr_outputs[0]
#             labels = torch.tensor([[i if i!=50256 else -100 for i in row] for row in curr_inputs_ids]).to(DEVICE)
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous(); # print(shift_logits.shape)
#             shift_labels = labels[..., 1:].contiguous(); # print(shift_labels.shape)        
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss(reduction='none')
#             loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
#             flat_losses.extend(loss.sum(axis=-1).cpu().detach().numpy().tolist())
#             ii += HFACE_BATCH_SIZE
#         except Exception as e:
#             # print(this_batch)
#             raise Exception(e)
# offset = 0
# batch_losses = []
# for val in lens:
#     batch_losses.append(flat_losses[offset:offset+val])
#     offset += val
# print(np.array(batch_losses))
# reranked_batch_predictions = [k_batch_predictions[np.argmin(batch_losses[i])][i] for i in range(len(batch_losses))]
# print(batch_clean_sentences)
# print("")
# print(reranked_batch_predictions)
# raise Exception("debug...")
# corr2corr_, corr2incorr_, incorr2corr_, incorr2incorr_ = \
#     get_metrics(batch_clean_sentences,batch_corrupt_sentences,reranked_batch_predictions,check_until_topk=1,return_mistakes=False)
# corr2corr+=corr2corr_
# corr2incorr+=corr2incorr_
# incorr2corr+=incorr2corr_
# incorr2incorr+=incorr2incorr_
##########################################################

# for i, (a,b,c,d) in enumerate(zip(batch_clean_sentences_,batch_corrupt_sentences_,reranked_batch_predictions,batch_predictions_k)):
#     if a==c: # right
#         line_index_right_opfile.write(f"{line_index+i}\t{a}\t{b}\t{c}\n")
#     else:
#         line_index_wrong_opfile.write(f"{line_index+i}\t{a}\t{b}\t{c}\n")
# line_index+=len(batch_clean_sentences_)
# line_index_right_opfile.flush()
# line_index_wrong_opfile.flush()

#  __mistakes = []
# __inds = []
# for i in range(len(batch_clean_sentences)):
#     if batch_clean_sentences[i].strip()!=k_batch_predictions[0][i].strip():
#         __mistakes.append(f"{batch_clean_sentences[i]}\n")
#         __inds.append(i)
# for k in range(topk):
#     batch_predictions_probs = k_batch_predictions_probs[k]
#     ii = 0
#     for ind in __inds:
#         __mistakes[ii]+=f"{batch_predictions_probs[ind]:.4f}\t"
#         ii+=1                
#     batch_predictions = k_batch_predictions[k]
#     ii = 0
#     for ind in __inds:
#         __mistakes[ii]+=f"{batch_predictions[ind]}\n"
#         ii+=1 
# ii=0
# for i,_ in enumerate(batch_clean_sentences):
#     if i in __inds:
#         __mistakes[ii]+="\n"
#         ii+=1
# for mis in __mistakes:
#     k_wrong_opfile.write(mis)

# __predictions = []
# for sent in batch_clean_sentences:
#     __predictions.append(f"{sent}\n")
# for k in range(topk):
#     batch_predictions_probs = k_batch_predictions_probs[k]
#     for i,val in enumerate(batch_predictions_probs):
#         __predictions[i]+=f"{val:.4f}\t"
#     batch_predictions = k_batch_predictions[k]
#     for i,sent in enumerate(batch_predictions):
#         __predictions[i]+=f"{sent}\n"
# for i,_ in enumerate(batch_clean_sentences):
#     __predictions[i]+="\n"
# for pred in __predictions:
#     k_right_opfile.write(pred)

# if beam_search:
#     line_index_right_opfile.close()
#     line_index_wrong_opfile.close()
#     k_wrong_opfile.close()
#     k_right_opfile.close()
# else:
#     line_index_right_opfile.close()
#     line_index_wrong_opfile.close()