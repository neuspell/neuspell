from typing import Set, Dict
import numpy as np
from typing import List
import json

from .candidate import Candidate




def edits_1(word: str) -> Set[str]:
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits_n(word: str, n: int, retval: Dict[int, Set[str]]) -> None:
    if n==1:
        retval[1] = edits_1(word)
        return

    # Recursive call
    edits_n(word, n-1, retval)

    # Store the current answer
    answer_n = set()
    edits_n_1 = retval[n-1]
    for word in edits_n_1:
        answer_n = answer_n.union(edits_1(word))
    
    # Save the current answer in retval
    retval[n] = answer_n









"""
Compute the Edit distance between two given
strings (s1 and s2)
"""

def get_edits_sub_del(str1: "original", str2: "noisy"):
    # find edit distane as usual and only pick selected edits from it
    len1, len2 = len(str1), len(str2)
    dp_dist = [[-1]*len2 for _ in range(len1)]
    dp_edits = [[[]]*len2 for _ in range(len1)]
    best_dist, best_edits = get_edits(str1, str2, len1-1, len2-1, dp_dist, dp_edits)
    # pick only deletions and substitutions of str1
    for edit in best_edits:
        if edit[-2]=="": 
            best_edits.remove(edit)
            best_dist-=1
    # return the selected edits and the modified editdist

    return best_dist, best_edits
def get_edits_add_sub_del(str1: "original", str2: "noisy"):
    # find edit distane as usual and only pick selected edits from it
    len1, len2 = len(str1), len(str2)
    dp_dist = [[-1]*len2 for _ in range(len1)]
    dp_edits = [[[]]*len2 for _ in range(len1)]
    best_dist, best_edits = get_edits(str1, str2, len1-1, len2-1, dp_dist, dp_edits)
    return best_dist, best_edits

def get_edits(str1, str2, ind1, ind2, dp_dist, dp_edits):
    if ind1==-1 and ind2==-1: return 0, []
    elif ind1==-1 and ind2>-1: return ind2+1, [(i,"",str2[i]) for i in range(ind2,-1,-1)]
    elif ind1>-1 and ind2==-1: return ind1+1, [(i,str1[i],"") for i in range(ind1,-1,-1)]
    if dp_dist[ind1][ind2]==-1:
        if str1[ind1]==str2[ind2]:
            return get_edits(str1, str2, ind1-1, ind2-1, dp_dist, dp_edits)
        else:
            # think and realize that changes are being made only to str1 as str2 is target
            # addition
            dist1, edits1 = get_edits(str1, str2, ind1, ind2-1, dp_dist, dp_edits)
            # substitution
            dist2, edits2 = get_edits(str1, str2, ind1-1, ind2-1, dp_dist, dp_edits)
            # deletion
            dist3, edits3 = get_edits(str1, str2, ind1-1, ind2, dp_dist, dp_edits)
            # pick least
            if dist3<=dist2 and dist3<=dist1:
                dp_dist[ind1][ind2], dp_edits[ind1][ind2] = \
                    dist3+1, edits3+[(ind1,str1[ind1],"")]
            elif dist2<=dist3 and dist2<=dist1:
                dp_dist[ind1][ind2], dp_edits[ind1][ind2] = \
                    dist2+1, edits2+[(ind1,str1[ind1],str2[ind2])]
            elif dist1<=dist2 and dist1<=dist3:
                dp_dist[ind1][ind2], dp_edits[ind1][ind2] = \
                    dist1+1, edits1+[(ind1+1,"",str2[ind2])]
    return dp_dist[ind1][ind2], dp_edits[ind1][ind2]
#print(get_edits_sub_del("champions","campionis"))
#print(get_edits_add_sub_del("champions","campionis"))







"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]