
import numpy as np
from tqdm import tqdm
# import json
# from typing import Set, Dict, List

from .candidate import Candidate
from .vocab import Vocab
from .edit_distance import edits_n, damerau_levenshtein_distance
from .double_metaphone import dm


class CandidatesGenerator(Vocab):
    """
    Vocab provides dictionary to query lexicon data as strings as well as token-ids
    CandidatesGenerator provides different (techniques) methods to generate candidates while using the above lexicon
    """
    
    def __init__(self, file_paths: list, do_meta = False):
        super(CandidatesGenerator, self).__init__(file_paths)
        self.do_meta = do_meta
        if self.do_meta:
            self.__make_doublemetaphones_dicts()
        # self.cache: keys are incorr words and values are retrieved 
        #   corr words at specified edit dist
        self.cache = {}
        self.max_dist = 1
        # self.edit_dist_cache: keys are tuples of word1 and word2, 
        #   value is the edit distance between them
        self.edit_dist_cache = {}

    def __make_doublemetaphones_dicts(self):
        self.token2dm = {}
        self.dm2token = {}
        # find dm for all tokens
        for token in tqdm(self.token2idx.keys()):
            if len(token.split())>1:
                raise Exception("token must be a single word or name")
            dm_cands = dm(token)
            self.token2dm[token] = [dm_cand for dm_cand in dm_cands if dm_cand]
            for dm_cand in dm_cands:
                if dm_cand not in self.dm2token:
                    self.dm2token[dm_cand] = []
                self.dm2token[dm_cand]+=[token]
        return

    def _get_editdistance_candidates(self, word: str, max_edit_dist: int, gamma=1.0):
        candidates = {}
        edits_n(word, max_edit_dist, candidates)
        """
        if word in self.token2idx:
            # adding same word at zero edit distance
            candidates[0] = set([word])
        """
        candidates[0] = set([word])
        candidate_set = {}
        for edit_dist in sorted(candidates.keys()):
            # edit_dist goes from zero to max_edit_dist
            word_set = candidates[edit_dist]
            for word in word_set:
                # select a word if in vocab and mask sure it is not duplicated in 
                #   the candidate_set
                if word in self.token2idx and word not in candidate_set:
                    new_candidate = Candidate(word, np.exp(-gamma * edit_dist))
                    candidate_set[word] = new_candidate
        return [*candidate_set.values()]

    def _get_doublemetaphone_candidates(self, word: str, max_dist: int):
        try:
            if self.token2dm=={} or self.dm2token=={}: 
                self.__make_doublemetaphones_dicts()
        except AttributeError:
            self.__make_doublemetaphones_dicts()
        # comparing function
        distbool = lambda str1,str2: str1==str2 if max_dist==0 else \
                        damerau_levenshtein_distance(str1,str2)<=max_dist
        # doublemetaphone of spelling mistake
        dm_word = dm(word)
        closest_dms = []
        for dm_cand in dm_word:
            closest_dms_ = [dm_ for dm_ in self.dm2token if (dm_cand!="" and distbool(dm_cand,dm_))]
            closest_dms += closest_dms_
        closest_tokens = [token for closest_dm in closest_dms for token in self.dm2token[closest_dm]]
        candidates = []
        for token in closest_tokens:
            cand = Candidate(token, np.exp(0))
            candidates+=[cand]
        return candidates

    def get_editdistance_doublemetaphone_candidates(self, word: str, max_dist: int = 1, do_meta=False):
        """
        retrieve from cache if available, careful with cached edit dists and curr max_dist
        else compute and save in cache
        """
        if max_dist!=self.max_dist:
            self.cache = {}  # reset
            self.max_dist = max_dist
        if word not in self.cache:
            cands = []
            cands += self._get_editdistance_candidates(word, max_dist)
            if do_meta:
                cands += self._get_doublemetaphone_candidates(word, 0)
            self.cache[word] = cands
        return self.cache[word]

    def get_edit_distance(self, str1: str, str2: str):
        return damerau_levenshtein_distance(str1,str2)
