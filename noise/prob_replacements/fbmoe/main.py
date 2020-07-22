

############################################
# README
# - commented snippets are for experimental purpose. 
#       Some of them are pre-ran and their data is stored.
# - stats dictionary with keys as following and value as the count of occurances:
#        [keys as context_length, i.e, numbers 0,1,2,3]
#            [keys as correct_character for each context length]
#                [keys as left context with start & end tokens appended (only whenever required)\
#                    to the context string, given the correct character]
#                    [keys as replacement characters for the obtained context]
#                        [values as counts]
#
# In the comments below, follow 1 to 3 steps for more details
############################################

import numpy as np
np.random.seed(11690)
import os
from tqdm import tqdm

from utils import noisyfy_backoff_homophones
from stats import load_stats, get_stats, save_stats, to_probs

STATS_JSON_PATH = 'moe_misspellings_train_ascii_stats_left_context.json'
NEW_STATS_JSON_PATH = 'moe_misspellings_train_ascii_stats_left_context.json'





if __name__=="__main__":

    ############################################
    # 1. load misspelling data and obtain stats
    ############################################


    # 1.1
    # load any stats already saved
    # stats is a dictionary as described above in NOTES
    existing_stats = load_stats(STATS_JSON_PATH)


    # 1.2
    # load some misspelling corpora
    # obtain moe_misspellings_train_ascii.tsv
    os.system("wget -O ./moe_misspellings_train.zip https://github.com/ \
        facebookresearch/moe/blob/master/data/moe_misspellings_train.zip?raw=true")
    os.system("unzip moe_misspellings_train.zip -d ./")
    linesWritten=0
    vocab = []
    isascii = lambda s: len(s) == len(s.encode())
    opfile = open("moe_misspellings_train.tsv","r")
    writefile1 = open("moe_misspellings_train_ascii.tsv","w")
    writefile2 = open("moe_misspellings_train_nonascii.tsv","w")
    lengths_no_match = 0
    for line in tqdm(opfile):
        try:
            w_e, w_c = line.strip().split("\t")
            l_e, l_c = len(w_e), len(w_c)
            if l_e<=l_c and isascii(w_c) and isascii(w_e):
                vocab.append(w_c);
                linesWritten+=1
                writefile1.write(w_e+"\t"+w_c+"\t"+str(l_e)+"\t"+str(l_c)+"\n")
                if l_e!=l_c: lengths_no_match+=1
            else:
                writefile2.write(line)
        except ValueError: # ValueError: not enough values to unpack (expected 2, got 1)
            writefile2.write(line)
            pass
    opfile.close()
    writefile1.close()
    writefile2.close()
    print("")
    print(f"lines written into new file are {linesWritten}")
    print(f"vocab length {len(set(vocab))}")
    print(f"lengths dont match for these many pairs {lengths_no_match}")


    # 1.3 
    # obtain more stats from  the new corpora and save them
    stats = get_stats("moe_misspellings_train_ascii.tsv", existing_stats, max_left_grams=3)
    save_stats(stats,NEW_STATS_JSON_PATH)


    # 1.4
    # convert counts to probabilities
    stats = to_probs(stats) 







    ############################################
    # 2. look at some noise injection examples
    ############################################


    """
    from utils import _get_replace_probs_all_contexts
    _get_replace_probs_all_contexts(
        stats, "", True, "o", alphas=[0.025, 0.05, 0.2, 0.7], print_stats=True)

    print(__get_replace_probs(stats, "<<hel", "l", 3,return_sorted_list=True))
    print(__get_replace_probs(stats, "el", "l", 2,return_sorted_list=True))
    print(__get_replace_probs(stats, "l", "l", 1,return_sorted_list=True))
    print(__get_replace_probs(stats, "", "l", 0,return_sorted_list=True))

    _get_replace_probs_all_contexts(
        stats, "ell", False, "o", alphas=[0.25,0.25,0.25,0.25], print_stats=True)
    print(_get_replace_probs_all_contexts(
            stats, "ell", False, "o", alphas=[0.25,0.25,0.25,0.25], print_stats=False))
    print("\n\n")
    print("\n\n")
    _get_replace_probs_all_contexts(
        stats, "ell", False, "o", alphas=[0, 0, 0, 1], print_stats=True)
    print("\n\n")
    print("\n\n")
    _get_replace_probs_all_contexts(
        stats, "", True, "p", alphas=[0.25,0.25,0.25,0.25], print_stats=True)
    print(_get_replace_probs_all_contexts(
        stats, "", True, "p", alphas=[0.25,0.25,0.25,0.25], print_stats=False))
    """














    ############################################
    # 3. noisy-fy clean data corpus
    # 3.1.1 or 3.1.2 for downloading clean data corpus
    ############################################

    # 3.1
    """
    # just upload one or more files from 1-billion-word-language-modeling-benchmark-r13output
    """
    clean_files = [
        "news.en-00001-of-00100","news.en-00002-of-00100",
        "news.en-00005-of-00100","news.en-00099-of-00100"]

    # 3.2 load the downloaded data
    for file in clean_file:
        lines = []
        for line in open(file,'r'):
            lines.append(line.strip())

    clean_files_vocab = {}
    for line in lines:
        for word in line.split():
            try:
                clean_files_vocab[word]+=1
            except KeyError:
                clean_files_vocab[word]=1
    clean_files_vocab_set = set([*clean_files_vocab.keys()])


    # 3.3 obtain homophones dictionary
    homophones = {}
    opfile = open('homophones.txt','r')
    for line in opfile:
        w1, w2 = line.strip().split('\t')
        try:
            homophones[w1].append(w2)
        except KeyError:
            homophones[w1] = [w2]
    opfile.close()
    homophones_set = set([*homophones.keys()])


    # 3.4  % (percentage) of words which are homophones
    commons_set = list(clean_files_vocab_set & homophones_set)
    print(f"no.of commom words: {len(commons_set)}")
    n1, n2 = 0, 0
    for word in clean_files_vocab_set:
        n1+=clean_files_vocab[word]
        if word in homophones_set:
            n2+=clean_files_vocab[word]
    print(f"n1->{n1},n2->{n2},%->{100*n2/n1}")


    # 3.5 select some random subset of sentences/lines
    # theseInds = np.arange(1000)
    theseInds = np.random.choice(np.arange(0,len(lines)), 1000, replace=False)   
    theseSents = [lines[i] for i in theseInds]


    # 3.6 print examples corrupted data from the selected data corpus
    new_lines = noisyfy_backoff_homophones(
        stats, theseSents, [0, 0, 0, 1], homophones, -1, print_data=True)

    new_lines = noisyfy_backoff_homophones(
        stats, theseSents, [0, 0, 1, 0], homophones, -1, print_data=True)

    new_lines = noisyfy_backoff_homophones(
        stats, theseSents, [0, 1, 0, 0], homophones, -1, print_data=True)

    new_lines = noisyfy_backoff_homophones(
        stats, theseSents, [1, 0, 0, 0], homophones, -1, print_data=True)

    new_lines = noisyfy_backoff_homophones(
        stats, theseSents, [0.025, 0.05, 0.2, 0.7], homophones, -1, print_data=True)
    

    # 3.7 corrupt the selected data corpus and save results
    for file in clean_files:
        lines = []
        for line in open(file,'r'):
            lines.append(line.strip()) 
        opfile.close()

        new_lines = noisyfy_backoff_homophones(stats,lines,[0.025, 0.05, 0.2, 0.7],homophones,0)
        opfile = open(file+".noise","w")
        for line in tqdm(new_lines):
            opfile.write(line+"\n")
        opfile.close()