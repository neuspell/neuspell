import json
import string

import numpy as np
from tqdm.autonotebook import tqdm

append_left_start = lambda context: "<<" + context
append_right_end = lambda context: context + ">>"
isascii = lambda s: len(s) == len(s.encode())
ispunct = lambda s: s in string.punctuation


def load_stats(file_name):
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
            stats[int(key)] = stats.pop(key)
    except:
        pass
    return stats


def get_lcs(str1: "correct word", str2: "error word"):
    l1, l2 = len(str1), len(str2)
    dp_counts = [[-np.inf] * l2 for _ in range(l1)]
    dp_changes = [[[]] * l2 for _ in range(l1)]
    return __get_dp(str1, str2, l1 - 1, l2 - 1, dp_counts, dp_changes)


def __get_dp(w_c, w_e, i_c, i_e, dp_counts, dp_changes):
    if not i_c >= i_e: return [], np.inf
    if i_e == -1 and i_c == -1: return [], 0
    if i_e == -1 and i_c >= 0: return [(i, w_c[i], "") for i in range(i_c, -1, -1)], i_c + 1

    if dp_counts[i_c][i_e] == -np.inf:  # need to fill
        if w_c[i_c] == w_e[i_e]:
            dp_changes[i_c][i_e], dp_counts[i_c][i_e] = \
                __get_dp(w_c, w_e, i_c - 1, i_e - 1, dp_counts, dp_changes)
        else:  # replace with char or replace will null in w_c as l_c>=l_e
            case1_changes, case1_count = __get_dp(w_c, w_e, i_c - 1, i_e - 1, dp_counts, dp_changes)
            case2_changes, case2_count = __get_dp(w_c, w_e, i_c - 1, i_e, dp_counts, dp_changes)
            if case1_count <= case2_count:
                dp_changes[i_c][i_e], dp_counts[i_c][i_e] = \
                    case1_changes + [(i_c, w_c[i_c], w_e[i_e])], case1_count + 1
            else:
                dp_changes[i_c][i_e], dp_counts[i_c][i_e] = \
                    case2_changes + [(i_c, w_c[i_c], "")], case2_count + 1
    return dp_changes[i_c][i_e], dp_counts[i_c][i_e]


def __get_replace_probs(stats, context_end_tokens, correct_char,
                        context_length_category, return_sorted_list=False):
    try:
        prob = stats[context_length_category][correct_char][context_end_tokens]
        if return_sorted_list:
            prob = sorted(prob.items(), key=operator.itemgetter(1), reverse=True)
    except KeyError:
        prob = {}
    return prob


def __sum_to_one(vals_):
    try:
        vals = vals_.copy()
        divide_by = sum(vals)
        return [val / divide_by for val in vals]
    except TypeError as e:
        print(vals_)
        raise Exception(e)


def __replace_only_topk(true_chars, mod_chars, mod_probs, top_k):
    return_nchanges = 0
    if top_k == 0: return "".join(true_chars), return_nchanges

    assert len(true_chars) == len(mod_chars) == len(mod_probs)
    _mod_inds, _mod_probs = [], []
    for i, (true_char, mod_char, mod_prob) in enumerate(zip(true_chars, mod_chars, mod_probs)):
        if mod_char != true_char: _mod_inds.append(i); _mod_probs.append(mod_prob)
    if not len(_mod_inds) == 0:
        for ind_ in np.random.choice(_mod_inds, min(top_k, len(_mod_inds)), \
                                     replace=False, p=__sum_to_one(_mod_probs)):
            true_chars[ind_] = mod_chars[ind_]
        return_nchanges += min(top_k, len(_mod_inds))
    return "".join(true_chars), return_nchanges


def _get_replace_probs_all_contexts(stats,
                                    raw_context,
                                    is_beginning,
                                    correct_char,
                                    alphas,
                                    print_stats=False,
                                    nascii=128):
    """
    # for a given context of length 0 to 3, and the character that has to be replaced
    #   this method, obtains a an ordered list of possiblereplacements, ordered in decreasing
    #   probability scores
    """
    # the first 0-127 ASCII include punctuations, numbers and alphabets
    assert len(raw_context) <= len(alphas) - 1

    replace_char_probs = [0] * (nascii + 1)
    epsilon_index = nascii

    sum_alpha = 0
    for ln in range(0, len(raw_context) + 1):  # [0,1,2,3,...,len(context)]
        selected_raw_context = raw_context[ln:]
        selected_raw_context_len = len(selected_raw_context)
        alpha = alphas[len(selected_raw_context)]
        if alpha != 0:
            sum_alpha += alpha
            selected_raw_context_new = append_left_start(selected_raw_context) \
                if (ln == 0 and is_beginning) else selected_raw_context
            probs_dict = __get_replace_probs(stats, selected_raw_context_new,
                                             correct_char, selected_raw_context_len)
            if print_stats: print(sorted(probs_dict.items(), key=operator.itemgetter(1), reverse=True))
            for replace_char, prob in probs_dict.items():
                if replace_char == "":
                    replace_char_probs[epsilon_index] += prob * alpha
                else:
                    replace_char_probs[ord(replace_char)] += prob * alpha
    if print_stats: print(f"sum_alpha: {sum_alpha}")
    normalize_by = sum_alpha
    if sum(replace_char_probs) == 0:
        if correct_char != "":
            replace_char_probs[ord(correct_char)] = 1
        else:
            replace_char_probs[epsilon_index] = 1
        normalize_by = 1
    else:
        replace_char_probs = [val / normalize_by for val in replace_char_probs]
    # should result in replace_char_probs s.t. sum(replace_char_probs)=1
    # but in cases where alpha is non-zero but probs are {}, the sum of probs is zero for that alpha,
    #       breaking the realization that
    #       << alpha1*[...]+alpha2*[....]+alpha3*[...]  / alpha1+alpha2+alpha3  = 1 >>
    replace_char_probs = __sum_to_one(replace_char_probs)
    if print_stats:
        replace_dict = {}
        for i, val in enumerate(replace_char_probs):
            if val != 0:
                if (i == nascii):
                    replace_dict[""] = val
                else:
                    replace_dict[chr(i)] = val
        replace_dict_sorted = sorted(replace_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(replace_dict_sorted)
        replace_me_with = []
        for _ in range(20):
            replace_char = np.random.choice([chr(p) \
                                             for p in range(nascii)] + [""], p=replace_char_probs)
            replace_me_with.append(replace_char)
        print(replace_me_with)
        return
    else:
        replace_char = np.random.choice([chr(p) for p in range(nascii)] + [""], p=replace_char_probs)
        replace_char_prob = replace_char_probs[ord(replace_char)] \
            if replace_char != "" else replace_char_probs[epsilon_index]
    return replace_char, replace_char_prob


def noisyfy_backoff_homophones(stats, lines, alphas, homophones={}, topk=-1, lower=False, print_data=False):
    """
    # noisy-fy some sampled clean data using previously computed stats
    #   using backoff weights (the previously computed stats dictionary) and
    #   using homophones (a dictionary of words with values as list of its homophones in english)
    # inputs
    #   stats: a dictionary of replacement probabilities (See stats.__add_this_info() for details)
    #   lines: a list of clean lines of text
    #   alphas: weightages for probability scores;
    #           index 0 to 3 correspondingly for 0 to 3 length context
    #   homophones: a dictionary of words as keys, each  value being a list of corresponding
    #               homophones
    #   topk: -1 to not use this arument, else choose the top-k most probable replacements
    #   lower: whether to lower case the clean data before injecting noise
    #   print_data: controls verbosity of the print statements; set to True for progress
    # outputs
    #   a list of corrupted lines, one line per line of input lines
    """
    max_gram = len(alphas) - 1
    homophones_set = set([*homophones.keys()])
    new_lines = []
    nchars, nchanges = 0, 0
    nwords, nwchanges = 0, 0
    print("total lines in inp to noisyfy_backoff_homophones: {}".format(len(lines)))
    print("total tokens in inp to noisyfy_backoff_homophones: {}".format(
        sum([len(line.strip().split()) for line in lines])))
    for lll, line in tqdm(enumerate(lines)):
        wtokens_ = line.strip().lower().split() if lower else line.strip().split()
        wtokens = line.strip().lower().split() if lower else line.strip().split()
        for ttt, token in enumerate(wtokens):
            nchars += len(token)
            if token in homophones_set and np.random.rand() <= 0.1:
                choices = homophones[token]  # obtain the choice list
                wtokens[ttt] = np.random.choice(choices)
                if print_data: print(lll, token, wtokens[ttt], " <-- homophone")
                _nchanges = get_lcs(token, wtokens[ttt])[1] if len(token) >= len(wtokens[ttt]) \
                    else get_lcs(wtokens[ttt], token)[1]
                nchanges += _nchanges
                continue
            top_k = topk
            if top_k == 0:
                if len(token) <= 3:
                    top_k = np.random.choice([0, 1], p=[0.9, 0.1])
                elif 3 < len(token) <= 6:
                    top_k = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
                else:
                    top_k = np.random.choice([0, 1, 2, 3, 4], p=[0.20, 0.25, 0.35, 0.15, 0.05])
            true_chars, mod_chars, mod_probs = [char for char in token], [], []
            _nchanges = 0
            for i, char in enumerate(token):
                if (not isascii(char)) or ispunct(char):
                    mod_chars += [char]
                    mod_probs += [0]
                    continue
                start_ = max(0, i - max_gram)
                replace_char, replace_char_prob = \
                    _get_replace_probs_all_contexts(stats, token[start_:i], \
                                                    True if start_ == 0 else False, token[i], alphas, print_stats=False)
                mod_chars.append(replace_char)
                mod_probs.append(replace_char_prob)
                if char != replace_char: _nchanges += 1
            if top_k >= 0:
                replace_token, _nchanges = __replace_only_topk(true_chars, mod_chars, mod_probs, top_k)
            else:
                replace_token, _nchanges = "".join(mod_chars), _nchanges
            nchanges += _nchanges
            wtokens[ttt] = replace_token
            if print_data and replace_token != token:
                print(lll, token, wtokens[ttt], " <-- corrupted", f" <-- top_{top_k}" if top_k >= 0 else "")
            if print_data and replace_token == token:
                print(lll, token, wtokens[ttt])
        new_lines.append(" ".join(wtokens))
        nwchanges += sum([1 if a != b else 0 for a, b in zip(wtokens_, wtokens)])
        nwords += len(wtokens_)
        if print_data: print("original line --> ", line)
        if print_data: print("corrupted line --> ", new_lines[lll])

    if print_data:
        print(f"total observed characters: {nchars}, \
               total corrupted characters: {nchanges}, \
               percent corrupted: {100 * nchanges / nchars}")
        print(f"total observed words: {nwords}, \
               total corrupted words: {nwchanges}, \
               percent corrupted: {100 * nwchanges / nwords}")

    return new_lines
