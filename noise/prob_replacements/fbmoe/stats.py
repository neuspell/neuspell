
############################################
# Methods to compute character replacement
#    probabilities based on left (+- right) 
#    context 
# (NEW!)  methods to save and use already 
#   computed stats
############################################

############################################
# References:
# - Spelling Correction and the Noisy Channel
#       https://web.stanford.edu/~jurafsky/slp3/B.pdf
# - Misspelling Oblivious Word Embeddings, Edizel et al. 2019
#       https://arxiv.org/abs/1905.09755
############################################

############################################
# Completed:
# - Obtaining the probabilities of replacement;
#       considering deletions as replacements 
#       with epsilon
# - Modify MOE paper's algo to chose top-K mistakes
# - Including homophones replacement
# - Implementing a code to run on all sentences of 1B data-file
############################################

############################################
# To do:
# - make dictionary of stats a new class instead ofdictionary
# - copy lines from old code to support right context inclusion
# - Smoothing for probability scores
# - Transpositions of characters
# - Phoneme replacement
# - Extend to other languages 
#       currently for 0-128 ASCII values only
############################################



import numpy as np
np.random.seed(11690)
import os
from tqdm import tqdm
import json

from utils import append_left_start, append_right_end
from utils import get_lcs 



def get_stats(file_name = "moe_misspellings_train_ascii.tsv", existing_stats={}, max_left_grams=3):
    """
    # inputs
        file_name : a new file for computing more statistics
        existing_stats : an already existing dictionary of counts
        max_left_grams : maximum context length to consider
    # outputs
    #   a dictionary of stats 
    """
    stats = existing_stats
    max_gram = max_left_grams
    total_entries = 0
    marked_entries = 0
    word_min_size = 1
    opfile = open(file_name,"r")
    for line in tqdm(opfile):
        total_entries+=1
        #if total_entries==1000000: break
        w_e, w_c, l_e, l_c = line.strip().split("\t")
        l_e, l_c = int(l_e), int(l_c)
        if l_c<word_min_size:
            continue

        if l_e==l_c:
            i_e, i_c = 0, 0
            while i_c<l_c and i_e<l_e:
                correct_char = w_c[i_c]
                if correct_char==w_e[i_e]: # not modified
                    for start_ in np.arange(max(0,i_c-max_gram),i_c+1,1):
                        left_context = w_c[start_:i_c]
                        if start_==0: left_context = append_left_start(left_context);
                        stats = __add_this_info(stats, left_context, 
                                                correct_char, correct_char, i_c-start_)
                    i_c+=1; i_e+=1;
                else: # corrupted with a different char             
                    modified_char = w_e[i_e]                    
                    for start_ in np.arange(max(0,i_c-max_gram),i_c+1,1):
                        left_context = w_c[start_:i_c]
                        if start_==0: left_context = append_left_start(left_context);
                        stats = __add_this_info(stats, left_context, 
                                                correct_char, modified_char, i_c-start_)                              
                    i_c+=1; i_e+=1;
        elif l_e<l_c:
            changes, changes_count = get_lcs(w_c,w_e)
            # each item of changes is a triplet of (index, original, corrupted_to)
            unchanged_indices = set(np.arange(l_c)) - set([change[0] for change in changes]) 
            for i_c in unchanged_indices:
                for start_ in np.arange(max(0,i_c-max_gram),i_c+1,1):
                    left_context = w_c[start_:i_c]
                    if start_==0: left_context = append_left_start(left_context);
                    stats = __add_this_info(stats, left_context, 
                                            correct_char, correct_char, i_c-start_)
            for i_c,org_char,mod_char in changes:
                for start_ in np.arange(max(0,i_c-max_gram),i_c+1,1):
                    left_context = w_c[start_:i_c]
                    if start_==0: left_context = append_left_start(left_context);
                    stats = __add_this_info(stats, left_context, 
                                            correct_char, mod_char, i_c-start_)
        else:
            marked_entries+=1
    opfile.close()
    print(f"total marked entries are {marked_entries} out of {total_entries}")
    return stats





def __add_this_info(stats, left_context, correct_char, modified_char, context_length_category):
    """
    # method to add items to the stats dictionary; a private method
    #
    stats dictionary, the heirarchy of keys are as follows:
        [keys as context_length, i.e, numbers 0,1,2,3]
            [keys as correct_character for each context length]
                [keys as left context with start & end tokens appended (only whenever required) \
                        to the context string, given the correct character]
                    [keys as replacement characters for the obtained context]
                        [values as counts]
    """
    if not (context_length_category in stats):
        stats[context_length_category] = {}
    if not (correct_char in stats[context_length_category]):
        stats[context_length_category][correct_char] = {}
    if not (left_context in stats[context_length_category][correct_char]):
        stats[context_length_category][correct_char][left_context] = {}
    if not (modified_char in stats[context_length_category][correct_char][left_context]):
        stats[context_length_category][correct_char][left_context][modified_char] = 0
    stats[context_length_category][correct_char][left_context][modified_char]+=1
    return stats





def save_stats(stats, file_name = 'moe_misspellings_train_ascii_stats_left_context.json'):
    """
    # keys converted to strings while saving
    """
    keys = [*stats.keys()]
    for key in keys: stats[str(key)]=stats.pop(key)
    
    opfile = open(file_name, 'w')
    json.dump(stats, opfile, sort_keys=True, indent=4)
    opfile.close()

    keys = [*stats.keys()]
    for key in keys: stats[int(key)]=stats.pop(key)
    return 





def load_stats(file_name = 'moe_misspellings_train_ascii_stats_left_context.json'):
    """
    # keys converted back from strings while loading
    """
    stats = {}
    try:
        opfile = open(file_name, 'r')
        stats = json.load(opfile)
        opfile.close()

        keys = [*stats.keys()]
        for key in keys:
            stats[int(key)]=stats.pop(key)
    except:
        pass
    return stats




def to_probs(stats):
    """
    # converting previously computed counts into probability scores
    # if already available as scores between 0 and 1
    #   this code snippet normalizes them to sum to 1 as required
    """
    for context_length_category in tqdm(stats):
        for correct_char in stats[context_length_category]:
            for left_context in stats[context_length_category][correct_char]:
                count = 0
                for modified_char in stats[context_length_category][correct_char][left_context]:
                    count+=stats[context_length_category][correct_char][left_context][modified_char]
                for modified_char in stats[context_length_category][correct_char][left_context]:
                    stats[context_length_category][correct_char][left_context][modified_char]/=count
    return stats



def to_probs_smoothed(stats):
    """
    # to implement
    """
    return to_probs(stats)