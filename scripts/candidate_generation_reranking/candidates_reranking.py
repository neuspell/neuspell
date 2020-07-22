
import os
import torch
import numpy as np
from tqdm import tqdm
import itertools
from typing import List

from transformers import BertForMaskedLM, BertTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

from .candidate import Candidate as CandidateUnit

class ContextProbability(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        print('Using ' + str(self.device))

    def _softmax(self, values):
        values = np.asarray(values).reshape(-1)
        max_values = np.max(values)
        values = values-max_values
        exp_values = np.exp(values)
        softmax_scores = exp_values/exp_values.sum()
        return softmax_scores.tolist()

class ContextProbabilityBERTLM(ContextProbability):

    def __init__(self,pretrained_model_path="",**kwargs):
        super().__init__(**kwargs)
        self.__mask_token = "[MASK]"
        self.__cls_token = "[CLS]"
        self.__sep_token = "[SEP]"
        self.__model_class = BertForMaskedLM
        self.__tokenizer_class = BertTokenizer
        self.__pretrained_type = 'bert-base-uncased'
        self.model, self.tokenizer = self._load_bert_model(pretrained_model_path)
        self.vocab = self._get_bert_vocab()

    def _load_bert_model(self, path):
        try:
            assert os.path.exists(path)==True 
            model = self.__model_class.from_pretrained(path)
            tokenizer = self.__tokenizer_class.from_pretrained(path)
        except:
            print("Unable to load model and tokenizer data from local path: \
                 {}; Loading from huggingface site".format(path))
            model = self.__model_class.from_pretrained(self.__pretrained_type)
            tokenizer = self.__tokenizer_class.from_pretrained(self.__pretrained_type)
        model = model.to(self.device)
        return model, tokenizer

    def _get_bert_vocab(self):
        if self.tokenizer==None:
            self.model, self.tokenizer = self._load_bert_model()
        return self.tokenizer.vocab

    def _get_masked_sents(self,tokenized_sentences,indices,add_terminal_tokens=False):
        assert len(tokenized_sentences)==len(indices)
        masked_sents = []
        for (tsent,ind) in zip(tokenized_sentences,indices):
            tsent[ind] = self.__mask_token
            msent = "{} {} {}".format(self.__cls_token," ".join(tsent),self.__sep_token) \
                if add_terminal_tokens else " ".join(tsent)
            masked_sents.append(msent)
        return masked_sents

    def __addto_forward_pass_stacklist(self,wtsent,masked_sents,tsent_seq_len,tsent_mask_index,candidate_map,i,candidate,vocab_search_candidate):
        potential_msent = " ".join(wtsent)
        if potential_msent in masked_sents:
            candidate_map[i][candidate].append((masked_sents.index(potential_msent),self.vocab[vocab_search_candidate]))
        else:
            masked_sents.append(potential_msent)
            dummy1 = self.tokenizer.tokenize(potential_msent)
            tsent_mask_index.append(dummy1.index(self.__mask_token)) #returns first index of self.__mask_token if multiple exists
            tsent_seq_len.append(len(dummy1))
            candidate_map[i][candidate].append((len(masked_sents)-1,self.vocab[vocab_search_candidate]))
        return masked_sents,tsent_seq_len,tsent_mask_index,candidate_map

    def _get_bert_probabilities(self,tokenized_sentences,indices,candidates=[],max_seq_len=-1) -> "List[List[CandidateUnit]]":

        for i, wtsent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [self.__cls_token]+wtsent+[self.__sep_token]

        masked_sents, tsent_seq_len, tsent_mask_index = [], [], []
        if len(candidates)>0:
            assert len(candidates)==len(tokenized_sentences)==len(indices)
            candidate_map = {}
            for i, (wtsent, wtind, row) in enumerate(zip(tokenized_sentences,indices,candidates)):
                candidate_map[i] = {}
                for candidate in row:
                    candidate_map[i][candidate] = []
                    if candidate in self.vocab:
                        wtsent_copy = list(wtsent)
                        wtsent_copy[wtind] = self.__mask_token
                        masked_sents,tsent_seq_len,tsent_mask_index,candidate_map = \
                            self.__addto_forward_pass_stacklist(wtsent_copy,masked_sents,tsent_seq_len,tsent_mask_index,candidate_map,i,candidate,candidate)
                    else:
                        # if broken_candidates = [c1, c2, c3, c4], we have four potential_msents with different masking scheme
                        broken_candidates = self.tokenizer.tokenize(candidate)
                        n_broken = len(broken_candidates)
                        replace_with = [self.__mask_token]*n_broken
                        wtsent_copy = list(wtsent)
                        wtsent_copy[wtind] = replace_with
                        wtsent_copy = list(itertools.chain.from_iterable(itertools.repeat(x,1) if isinstance(x,str) else x for x in wtsent_copy))
                        masked_sents,tsent_seq_len,tsent_mask_index,candidate_map = \
                            self.__addto_forward_pass_stacklist(wtsent_copy,masked_sents,tsent_seq_len,tsent_mask_index,candidate_map,i,candidate,broken_candidates[0])
                        for temp_i, br_candidate in enumerate(broken_candidates[:-1]):
                            replace_with[temp_i] = br_candidate
                            wtsent_copy = list(wtsent)
                            wtsent_copy[wtind] = replace_with
                            wtsent_copy = list(itertools.chain.from_iterable(itertools.repeat(x,1) if isinstance(x,str) else x for x in wtsent_copy))
                            masked_sents,tsent_seq_len,tsent_mask_index,candidate_map = \
                                self.__addto_forward_pass_stacklist(wtsent_copy,masked_sents,tsent_seq_len,tsent_mask_index,candidate_map,i,candidate,broken_candidates[temp_i+1])
            return_selected_candidates = True
            #print(candidate_map)
            #print(masked_sents)
        else:
            assert len(tokenized_sentences)==len(indices)
            masked_sents = self._get_masked_sents(tokenized_sentences,indices,add_terminal_tokens=True)
            for i, msent in enumerate(masked_sents):
                tsent = self.tokenizer.tokenize(msent)
                tsent_seq_len.append(len(tsent))
                tsent_mask_index.append(tsent.index(self.__mask_token))
            return_selected_candidates = False

        nsents = len(masked_sents)
        
        BATCH_SIZE, batch_start = 128, 0
        # obtain softmax scores as list; if converted to numpy has shape [nsents,bert_vocab_size]
        softmax_scores = []
        while(batch_start<nsents):

            batch_masked_sents = masked_sents[batch_start:min(batch_start+BATCH_SIZE,nsents)]
            batch_tsent_seq_len = tsent_seq_len[batch_start:min(batch_start+BATCH_SIZE,nsents)]
            batch_tsent_mask_index = tsent_mask_index[batch_start:min(batch_start+BATCH_SIZE,nsents)]

            # max_seq_len must be at least 1 greater than until MASK index length!!
            if max_seq_len!=-1 and len([*filter(lambda x: x <= 0, max_seq_len-batch_tsent_seq_len-2)])==0:
                MAX_SEQUENCE_LEN = max_seq_len
            else:
                MAX_SEQUENCE_LEN = np.max(batch_tsent_seq_len)

            batch_encoded_sents = []
            for msent in batch_masked_sents:
                tsent = self.tokenizer.tokenize(msent)
                if len(tsent)>MAX_SEQUENCE_LEN:
                    tsent = tsent[:MAX_SEQUENCE_LEN]
                    tsent[-1] = ["[SEP]"]
                tsent += ["[PAD]"]*(MAX_SEQUENCE_LEN-len(tsent))
                idsent = self.tokenizer.convert_tokens_to_ids(tsent)
                batch_encoded_sents.append(idsent)
            batch_input_ids = torch.tensor(batch_encoded_sents)
            batch_input_ids = batch_input_ids.to(self.device)

            with torch.no_grad():
                batch_final_layer_logits = self.model(batch_input_ids)[0] #[BATCH_SIZE,MAX_SEQUENCE_LEN,32k]

            batch_final_layer_logits = batch_final_layer_logits.to('cpu')
            for _final_layer_logits, _tsent_mask_index in zip(batch_final_layer_logits,batch_tsent_mask_index):
                curr_softmax_scores = self._softmax(_final_layer_logits[_tsent_mask_index,:].reshape(-1))
                softmax_scores.append(curr_softmax_scores)

            batch_start+=BATCH_SIZE

        
        if return_selected_candidates:
            return_candidate_instances = []
            for i in candidate_map.keys():
                curr_candidate_instances = []
                for candidate in candidate_map[i].keys():
                    # can have a list of posibilities as [(x1,y1),...,(xn,yn)]
                    #   due to split into sub word units
                    prod = 1
                    for (ind1,ind2) in candidate_map[i][candidate]:
                        prod*=softmax_scores[ind1][ind2]
                    curr_candidate_instances.append(CandidateUnit(candidate,prod))
                return_candidate_instances.append(curr_candidate_instances)
        else:
            return_candidate_instances = []
            for i in range(0,nsents):
                curr_candidate_instances = []
                for (candidate, score) in zip([*self.vocab.keys()],softmax_scores[i]):
                    curr_candidate_instances.append(CandidateUnit(candidate,score))
                return_candidate_instances.append(curr_candidate_instances)

        # sort
        for i, candidate_list in enumerate(return_candidate_instances):
            return_candidate_instances[i] = sorted(candidate_list, key=lambda unit_:unit_.prob, reverse=True)

        return return_candidate_instances

class SentenceProbability:
    def __init__(self, device='cpu'):
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        model = model.to(device)

        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device

        self.NUM_CLASSES = 267735
    
    def get_sentence_probabilities(self, sentences: List[List[str]]) -> torch.Tensor:
        """
        Given a list of list of words (list of sentences), this function computes the
        probability of the complete sentence according to the TransformerXL model.

        Returns a 1-D tensor with probability values for each sentence.
        """
        sentences = sorted(sentences, key=lambda sentence: len(sentence))
        
        input_ids = self.tokenizer.encode_sents(sentences)
        input_mask = [torch.ones_like(input_tensor) for input_tensor in input_ids]
        
        padded_input_tensor = pad_sequence(input_ids, batch_first=True).to(self.device)
        padded_mask_tensor = pad_sequence(input_mask, batch_first=True).float().to(self.device)

        with torch.no_grad():
            prediction_scores, _ = self.model(padded_input_tensor)[:2]

        expanded_input_tensor = one_hot(padded_input_tensor, num_classes=self.NUM_CLASSES).float()
        word_wise_probabilities = torch.mul(expanded_input_tensor, prediction_scores).sum(dim=2)

        masked_word_wise_probabilities = torch.mul(word_wise_probabilities, padded_mask_tensor)
        sentence_wise_probabilities = masked_word_wise_probabilities.sum(dim=1)

        return sentence_wise_probabilities.to('cpu')

    def get_candidate_probabilities(self,
        sentences: List[List[str]],
        indices: List[int],
        candidates: List[List[str]]) -> torch.Tensor:
        """
        Given sentences, indices for each sentence, and potential candidates
        for those indices, this function returns a 2D tensor containing
        probabilities for each sentence and candidate pair in the same order.

        Assumption - All words have equal number of candidates.
        """
        all_probabilities = []
        for sentence, index, candidate_list in zip(sentences, indices, candidates):
            candidate_sentence_list = []
            for candidate in candidate_list:
                new_sentence = sentence.copy()
                new_sentence[index] = candidate
                candidate_sentence_list.append(new_sentence)
            
            all_probabilities.append(self.get_sentence_probabilities(candidate_sentence_list))

        return torch.stack(all_probabilities)










if __name__=="__main__":

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    contextProbability = ContextProbabilityBERTLM(device=device)
    tokenized_sentences =   [
                            ["he","ran","in","the","prak"],
                            ["when","was","lest","tme","you","wrte","to","me"],
                            ["te","envronment","is","geting","poluted"],
                            ["te","envronment","is","geting","poluted"],
                            ["te","envronment","is","geting","poluted"],
                            ["te","envronment","is","geting","poluted"],
                            ["te","envronment","is","geting","poluted"]
                            ]
    indices = [4,2,0,1,2,3,4]
    candidates =    [
                    ["prka","mrak","park","prank","peak","plan","pork"],
                    ["lyst","lezt","list","last","less","lust","lost","test","nest","best"],
                    ["t","we","me","th","the","teh"],
                    ["envros","envirooonnn","envos","environmment","environment"],
                    ["was","ws","as","us"],
                    ["geing","geeting","getting","getiing"],
                    ["pouuted","polluted","pollented"]
                    ]
    print(contextProbability._get_bert_probabilities(tokenized_sentences,indices,candidates))