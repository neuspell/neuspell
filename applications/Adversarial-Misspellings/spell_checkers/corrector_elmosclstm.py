
###################################
# USAGE
# -----
# cd ./src/correctors
# python
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from corrector_elmosclstm import CorrectorElmoSCLstm
correctorElmoSCLstm = CorrectorElmoSCLstm()
correctorElmoSCLstm.correct_strings(["Hellow Wald", "They fought a deadly wear !!"])
correctorElmoSCLstm.correct_strings(["An illustrative example of noised text","An isulvtriatle epaxmle of nsieod txet","An illstrative examle of nosed tet","n ilelustrative edxample of nmoised texut","An ilkustrative exsmple of npised test"])
'''
###################################


import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")

import torch
from typing import List

from scripts.seq_modeling.elmosclstm import load_model, load_vocab_dict, load_pretrained, model_predictions

class CorrectorElmoSCLstm(object):
    
    def __init__(self, DATA_FOLDER_PATH="../../data", backoff="pass-through"):

        print(f"backoff: {backoff}")

        BASE_PATH = DATA_FOLDER_PATH
        # none #probnoise #wordnoise #random #probwordnoise #probwordnoise-moviereviewsfinetune #probwordnoise-moviereviewsfinetune2
        CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints/elmoscrnn-probwordnoise-moviereviewsfinetune") 
        print(f"in CorrectorElmoSCLstm, loading model from CHECKPOINT_PATH:{CHECKPOINT_PATH}")
        VOCAB_PATH = os.path.join(CHECKPOINT_PATH,"vocab.pkl")
        print(f"loading vocab from {VOCAB_PATH}")
        self.vocab = load_vocab_dict(VOCAB_PATH)

        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = load_model(self.vocab)
        self.model = load_pretrained(self.model,CHECKPOINT_PATH)
        self.backoff = backoff

    def correct_string(self, mystring: str) -> str:
        return self.correct_strings([mystring])[0]

    def correct_strings(self, mystrings: List[str]) -> List[str]:
        data = [(line,line) for line in mystrings]
        return_strings = model_predictions(self.model, data, self.vocab, DEVICE=self.DEVICE, BATCH_SIZE=16, backoff=self.backoff)
        return return_strings
