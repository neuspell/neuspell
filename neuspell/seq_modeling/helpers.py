import os
import pickle
import sys
from math import log

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def progressBar(value, endvalue, names, values, bar_length=30):
    assert (len(names) == len(values));
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow));
    string = '';
    for name, val in zip(names, values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val != None else '|| {0}: {1} '.format(name, None)
        string += temp;
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    return


def load_data(base_path, corr_file, incorr_file):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r")
    for line in opfile1:
        if line.strip() != "": incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r")
    for line in opfile2:
        if line.strip() != "": corr_data.append(line.strip())
    opfile2.close()
    assert len(incorr_data) == len(corr_data)

    # verify if token split is same
    for i, (x, y) in tqdm(enumerate(zip(corr_data, incorr_data))):
        x_split, y_split = x.split(), y.split()
        try:
            assert len(x_split) == len(y_split)
        except AssertionError:
            print("# tokens in corr and incorr mismatch. retaining and trimming to min len.")
            print(x_split, y_split)
            mn = min([len(x_split), len(y_split)])
            corr_data[i] = " ".join(x_split[:mn])
            incorr_data[i] = " ".join(y_split[:mn])
            print(corr_data[i], incorr_data[i])

    # return as pairs
    data = []
    for x, y in tqdm(zip(corr_data, incorr_data)):
        data.append((x, y))

    print(f"loaded tuples of (corr,incorr) examples from {base_path}")
    return data


def train_validation_split(data, train_ratio, seed):
    np.random.seed(seed)
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio * len_))
    inds_shuffled = np.arange(len_);
    np.random.shuffle(inds_shuffled);
    train_data = []
    for ind in inds_shuffled[:train_len_]: train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]: validation_data.append(data[ind])
    return train_data, validation_data


def get_char_tokens(use_default: bool, data=None):
    if not use_default and data is None: raise Exception("data is None")

    # reset char token utils
    chartoken2idx, idx2chartoken = {}, {}
    char_unk_token, char_pad_token, char_start_token, char_end_token = \
        "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>"
    special_tokens = [char_unk_token, char_pad_token, char_start_token, char_end_token]
    for char in special_tokens:
        idx = len(chartoken2idx)
        chartoken2idx[char] = idx
        idx2chartoken[idx] = char

    if use_default:
        chars = len(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))
        for char in chars:
            if char not in chartoken2idx:
                idx = len(chartoken2idx)
                chartoken2idx[char] = idx
                idx2chartoken[idx] = char
    else:
        # helper funcs
        # isascii = lambda s: len(s) == len(s.encode())
        """
        # load batches of lines and obtain unique chars
        nlines = len(data)
        bsize = 5000
        nbatches = int( np.ceil(nlines/bsize) )
        for i in tqdm(range(nbatches)):
            blines = " ".join( [ex for ex in data[i*bsize:(i+1)*bsize]] )
            #bchars = set(list(blines))
            for char in bchars:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char
        """
        # realized that set doesn't preserve order!!
        for line in tqdm(data):
            for char in line:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char

    print(f"number of unique chars found: {len(chartoken2idx)}")
    print(chartoken2idx)
    return_dict = {}
    return_dict["chartoken2idx"] = chartoken2idx
    return_dict["idx2chartoken"] = idx2chartoken
    return_dict["char_unk_token"] = char_unk_token
    return_dict["char_pad_token"] = char_pad_token
    return_dict["char_start_token"] = char_start_token
    return_dict["char_end_token"] = char_end_token
    # new
    return_dict["char_unk_token_idx"] = chartoken2idx[char_unk_token]
    return_dict["char_pad_token_idx"] = chartoken2idx[char_pad_token]
    return_dict["char_start_token_idx"] = chartoken2idx[char_start_token]
    return_dict["char_end_token_idx"] = chartoken2idx[char_end_token]

    return return_dict


def get_tokens(data,
               keep_simple=False,
               min_max_freq=(1, float("inf")),
               topk=None,
               intersect=[],
               load_char_tokens=False):
    # get all tokens
    token_freq, token2idx, idx2token = {}, {}, {}
    for example in tqdm(data):
        for token in example.split():
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
    print(f"Total tokens found: {len(token_freq)}")

    # retain only simple tokens
    if keep_simple:
        isascii = lambda s: len(s) == len(s.encode())
        hasdigits = lambda s: len([x for x in list(s) if x.isdigit()]) > 0
        tf = [(t, f) for t, f in [*token_freq.items()] if (isascii(t) and not hasdigits(t))]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only tokens with specified min and max range
    if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        tf = [(i[0], i[1]) for i in sorted_ if (i[1] >= min_max_freq[0] and i[1] <= min_max_freq[1])]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only topk tokens
    if topk is not None:
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only interection of tokens
    if len(intersect) > 0:
        tf = [(t, f) for t, f in [*token_freq.items()] if (t in intersect or t.lower() in intersect)]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # create token2idx and idx2token
    for token in token_freq:
        idx = len(token2idx)
        idx2token[idx] = token
        token2idx[token] = idx

    # add <<PAD>> special token
    ntokens = len(token2idx)
    pad_token = "<<PAD>>"
    token_freq.update({pad_token: -1})
    token2idx.update({pad_token: ntokens})
    idx2token.update({ntokens: pad_token})

    # add <<UNK>> special token
    ntokens = len(token2idx)
    unk_token = "<<UNK>>"
    token_freq.update({unk_token: -1})
    token2idx.update({unk_token: ntokens})
    idx2token.update({ntokens: unk_token})

    # new
    # add <<EOS>> special token
    ntokens = len(token2idx)
    eos_token = "<<EOS>>"
    token_freq.update({eos_token: -1})
    token2idx.update({eos_token: ntokens})
    idx2token.update({ntokens: eos_token})

    # return dict
    token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
    return_dict = {"token2idx": token2idx,
                   "idx2token": idx2token,
                   "token_freq": token_freq,
                   "pad_token": pad_token,
                   "unk_token": unk_token,
                   "eos_token": eos_token
                   }
    # new
    return_dict.update({
        "pad_token_idx": token2idx[pad_token],
        "unk_token_idx": token2idx[unk_token],
        "eos_token_idx": token2idx[eos_token],
    })

    # load_char_tokens
    if load_char_tokens:
        print("loading character tokens")
        char_return_dict = get_char_tokens(use_default=False, data=data)
        return_dict.update(char_return_dict)

    return return_dict


def num_unk_tokens(sentences, vocab):
    token2idx = vocab["token2idx"]
    sum_ = 0
    total_ = 0
    for line in tqdm(sentences):
        sum_ += sum([1 if token not in token2idx else 0 for token in line.strip().split()])
        total_ += len(line.strip().split())
    print("#unk-tokenizations: {}/{}, %unk-tokenizations: {:.4f}".format(sum_, total_, 100 * sum_ / total_))
    return


# train utils

def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of lables and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:  np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch_labels = [data[idx][0] for idx in batch_indices]
        batch_sentences = [data[idx][1] for idx in batch_indices]

        yield (batch_labels, batch_sentences)


def labelize(batch_labels, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_labels]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def tokenize(batch_sentences, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_sentences]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def char_tokenize(batch_sentences, vocab, return_nchars=False):
    """
    return (List[pad_sequence],Tensor[int]) if as_tensor=True
    return (List[List[int]],List[int],List[int]) if as_tensor=False
    """
    as_tensor = True

    chartoken2idx = vocab["chartoken2idx"]
    char_unk_token = vocab["char_unk_token"]
    char_pad_token = vocab["char_pad_token"]
    char_start_token = vocab["char_start_token"]
    char_end_token = vocab["char_end_token"]

    func_word2charids = lambda word: [chartoken2idx[char_start_token]] + \
                                     [chartoken2idx[char] if char in chartoken2idx else chartoken2idx[char_unk_token] \
                                      for char in list(word)] + \
                                     [chartoken2idx[char_end_token]]

    if as_tensor:
        # char_padding_idx = chartoken2idx[char_pad_token]
        # tokenized_output = [ pad_sequence(
        #                             [torch.as_tensor(func_word2charids(word)).long() for word in sent.split()],
        #                             batch_first=True,
        #                             padding_value=char_padding_idx
        #                             ) \
        #                     for sent in batch_sentences]
        # nwords = torch.tensor([len(sentlevel) for sentlevel in tokenized_output]).long()
        # return tokenized_output, nwords        
        char_idxs = [[func_word2charids(word) for word in sent.split()] for sent in batch_sentences]
        char_padding_idx = chartoken2idx[char_pad_token]
        tokenized_output = [pad_sequence(
            [torch.as_tensor(list_of_wordidxs).long() for list_of_wordidxs in list_of_lists],
            batch_first=True,
            padding_value=char_padding_idx
        ) \
            for list_of_lists in char_idxs]
        # dim [nsentences]
        nwords = torch.tensor([len(sentlevel) for sentlevel in tokenized_output]).long()
        # dim [nsentences,nwords_per_sentence]
        nchars = [torch.as_tensor([len(wordlevel) for wordlevel in sentlevel]).long() for sentlevel in char_idxs]
    else:
        char_idxs = [[func_word2charids(word) for word in sent.split()] for sent in batch_sentences]
        tokenized_output = char_idxs
        # dim [nsentences]
        nwords = [len(sentlevel) for sentlevel in char_idxs]
        # dim [nsentences,nwords_per_sentence]
        nchars = [[len(wordlevel) for wordlevel in sentlevel] for sentlevel in char_idxs]
    # output
    if not return_nchars:
        return tokenized_output, nwords
    else:
        return tokenized_output, nwords, nchars


def sclstm_tokenize(batch_sentences, vocab):
    """
    return (List[pad_sequence],Tensor[int])
    """
    chartoken2idx = vocab["chartoken2idx"]
    char_unk_token_idx = vocab["chartoken2idx"][vocab["char_unk_token"]]

    def sc_vector(word):
        a = [0] * len(chartoken2idx)
        if word[0] in chartoken2idx:
            a[chartoken2idx[word[0]]] = 1
        else:
            a[char_unk_token_idx] = 1
        b = [0] * len(chartoken2idx)
        for char in word[1:-1]:
            if char in chartoken2idx: b[chartoken2idx[char]] += 1
            # else: b[ char_unk_token_idx ] = 1
        c = [0] * len(chartoken2idx)
        if word[-1] in chartoken2idx:
            c[chartoken2idx[word[-1]]] = 1
        else:
            c[char_unk_token_idx] = 1
        return a + b + c

    # return list of tesnors and we don't need to pad these unlike cnn-lstm case!
    tensor_output = [torch.tensor([sc_vector(word) for word in sent.split()]).float() for sent in batch_sentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in tensor_output]).long()
    return tensor_output, nwords


def sctrans_tokenize(batch_sentences, vocab):
    """
    return (List[pad_sequence],Tensor[int],Tensor[int])
    """
    chartoken2idx = vocab["chartoken2idx"]
    char_unk_token_idx = vocab["chartoken2idx"][vocab["char_unk_token"]]

    def sc_vector(word):
        a = [0] * len(chartoken2idx)
        if word[0] in chartoken2idx:
            a[chartoken2idx[word[0]]] = 1
        else:
            a[char_unk_token_idx] = 1
        b = [0] * len(chartoken2idx)
        for char in word[1:-1]:
            if char in chartoken2idx: b[chartoken2idx[char]] += 1
            # else: b[ char_unk_token_idx ] = 1
        c = [0] * len(chartoken2idx)
        if word[-1] in chartoken2idx:
            c[chartoken2idx[word[-1]]] = 1
        else:
            c[char_unk_token_idx] = 1
        return a + b + c

    # return list of tensors and we don't need to pad these unlike cnn-lstm case!
    # tensor_output =  [ torch.tensor([sc_vector(word) for word in sent.split()]).float() for sent in batch_sentences]
    # ------NEW---- Added +[0,0]
    tensor_output = [torch.tensor([sc_vector(word) + [0, 0] for word in sent.split()]).float() for sent in
                     batch_sentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in tensor_output]).long()
    inverted_mask = pad_sequence([torch.zeros(len_) for len_ in nwords], batch_first=True, padding_value=1).bool()
    return tensor_output, nwords, inverted_mask


def untokenize(batch_predictions, batch_lengths, vocab):
    idx2token = vocab["idx2token"]
    unktoken = vocab["unk_token"]
    assert len(batch_predictions) == len(batch_lengths)
    batch_predictions = \
        [" ".join([idx2token[idx] for idx in pred_[:len_]]) \
         for pred_, len_ in zip(batch_predictions, batch_lengths)]
    return batch_predictions


def untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff="pass-through"):
    assert backoff in ["neutral", "pass-through"], print(f"selected backoff strategy not implemented: {backoff}")
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]
    if backoff == "pass-through":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else clean_[i] for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    elif backoff == "neutral":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else "a" for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    return batch_predictions


def untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_clean_sentences, topk=None):
    """
    batch_predictions are softmax probabilities and should have shape (batch_size,max_seq_len,vocab_size)
    batch_lengths should have shape (batch_size)
    batch_clean_sentences should be strings of shape (batch_size)
    """
    # print(batch_predictions.shape)
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    if topk is not None:
        # get topk items from dim=2 i.e top 5 prob inds
        batch_predictions = np.argpartition(-batch_predictions, topk, axis=-1)[:, :,
                            :topk]  # (batch_size,max_seq_len,5)
    # else:
    #    batch_predictions = batch_predictions # already have the topk indices

    # get topk words
    idx_to_token = lambda idx, idx2token, corresponding_clean_token, unktoken: idx2token[
        idx] if idx != unktoken else corresponding_clean_token
    batch_predictions = \
        [[[idx_to_token(wordidx, idx2token, batch_clean_sentences[i][j], unktoken) \
           for wordidx in topk_wordidxs] \
          for j, topk_wordidxs in enumerate(predictions[:batch_lengths[i]])] \
         for i, predictions in enumerate(batch_predictions)]

    return batch_predictions


def untokenize_without_unks3(batch_predictions, batch_predictions_probs, batch_lengths, vocab, batch_clean_sentences,
                             topk):
    """
    batch_predictions are indices of vocab that have the top values at each timestep and should have shape (batch_size,max_seq_len,vocab_size)
    batch_predictions_probs are the corresponding softmaxed probabilities and should also have shape (batch_size,max_seq_len,vocab_size)
    batch_lengths should have shape (batch_size)
    batch_clean_sentences should be strings of shape (batch_size)
    topk is the beam size

    returns k_batch_predictions:
        k different lists, each list being a batch_predictions (list of strings)
    """
    assert batch_predictions.shape == batch_predictions_probs.shape
    assert topk is not None and topk > 1, print("topk argument cannot be None or less than equal to 1")

    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    k_batch_predictions = {i: [] for i in range(topk)}
    k_batch_predictions_probs = [[] for i in range(topk)]
    # (max_seq_len,vocab_size) and (1) dimensions
    for _prediction_probs, _length in zip(batch_predictions_probs, batch_lengths):
        sequences = beam_search_decoder(_prediction_probs[:_length], topk)
        for i in range(topk):
            k_batch_predictions_probs[i].append(sequences[i][1])
        k_batch_predictions = {i: k_batch_predictions[i] + [sequences[i][0]] for i in range(topk)}

    # for ri in range(len([*k_batch_predictions.values()][0])):
    #     print("\n\n")
    #     for k in range(topk):
    #         print([*k_batch_predictions.values()][k][ri])

    # commentme = \
    #     [
    #         [ [   idx if idx!=unktoken else "UNK" for i, idx in enumerate([p[ip] for p,ip in zip(pred_[:len_],pred_ind_[:len_])])  ] \
    #          for pred_,pred_ind_,len_,clean_ in zip(batch_predictions,batch_predictions_inds,batch_lengths,batch_clean_sentences) ]
    #         for _,batch_predictions_inds in k_batch_predictions.items()
    #     ]
    # print(commentme)

    # now, each value of k_batch_predictions can be considered as batch_predictions for
    #   the code re-usability purpose. each is of dims (batch_size,seq_len,topk) 
    #   seq_len is the corresponding batch sentence's length and not max_seq_len

    k_batch_predictions = \
        [
            [" ".join([idx2token[idx] if idx != unktoken else clean_[i] for i, idx in
                       enumerate([p[ip] for p, ip in zip(pred_[:len_], pred_ind_[:len_])])]) \
             for pred_, pred_ind_, len_, clean_ in
             zip(batch_predictions, batch_predictions_inds, batch_lengths, batch_clean_sentences)]
            for _, batch_predictions_inds in k_batch_predictions.items()
        ]
    # print(k_batch_predictions)
    # raise Exception("debug...")

    # print(k_batch_predictions[1])
    # print(batch_predictions)
    # print(k_batch_predictions[1])
    # raise Exception("debug...")

    return k_batch_predictions, k_batch_predictions_probs


def beam_search_decoder(data, k):
    """
    inputs:
        data: 2d array of softmax probability values; of shape (seq_len,vocab_size)
        k: the beam width
    outputs:
        sequences: a list of lists; k lists each consisting of sequence and log-sum-probability
                    ex: [
                         [[4, 0, 3, 5], -2.38],
                         [[4, 1, 4, 5], -3.64],
                         [[2, 0, 3, 3], -7.44]
                         ]
                    when k=3 and seq_len=4
    """
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # print(len(sequences))
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score + log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)  # descending
        # select k best
        sequences = ordered[:k]
    return sequences


def get_model_nparams(model):
    ntotal = 0
    for param in list(model.parameters()):
        temp = 1
        for sz in list(param.size()): temp *= sz
        ntotal += temp
    return ntotal


def batch_accuracy_func(batch_predictions: np.ndarray,
                        batch_targets: np.ndarray,
                        batch_lengths: list):
    """
    given the predicted word idxs, this method computes the accuracy 
    by matching all values from 0 index to batch_lengths_ index along each 
    batch example
    """
    assert len(batch_predictions) == len(batch_targets) == len(batch_lengths)
    count_ = 0
    total_ = 0
    for pred, targ, len_ in zip(batch_predictions, batch_targets, batch_lengths):
        count_ += (pred[:len_] == targ[:len_]).sum()
        total_ += len_
    return count_, total_


def load_vocab_dict(path_: str):
    """
    path_: path where the vocab pickle file is saved
    """
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab


def save_vocab_dict(path_: str, vocab_: dict):
    """
    path_: path where the vocab pickle file to be saved
    vocab_: the dict data
    """
    with open(path_, 'wb') as fp:
        pickle.dump(vocab_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


################################################
# ----->
# For BERT Custom Tokenization
################################################

import numpy as np
import transformers

# import torch
# from torch.nn.utils.rnn import pad_sequence
BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-cased')
BERT_TOKENIZER.do_basic_tokenize = True
BERT_TOKENIZER.tokenize_chinese_chars = False
BERT_MAX_SEQ_LEN = 512


def merge_subtokens(tokens: "list"):
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"):
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    text = " ".join(merged_tokens)
    return text


def _custom_bert_tokenize_sentence(text):
    tokens = BERT_TOKENIZER.tokenize(text)
    tokens = tokens[:BERT_MAX_SEQ_LEN - 2]  # 2 allowed for [CLS] and [SEP]
    idxs = np.array([idx for idx, token in enumerate(tokens) if not token.startswith("##")] + [len(tokens)])
    split_sizes = (idxs[1:] - idxs[0:-1]).tolist()
    # NOTE: BERT tokenizer does more than just splitting at whitespace and tokenizing. So be careful.
    # -----> assert len(split_sizes)==len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes, text)
    # -----> hence do the following:
    text = merge_subtokens(tokens)
    assert len(split_sizes) == len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes,
                                                        text)
    return text, tokens, split_sizes


def _custom_bert_tokenize_sentences(list_of_texts):
    out = [_custom_bert_tokenize_sentence(text) for text in list_of_texts]
    texts, tokens, split_sizes = list(zip(*out))
    return [*texts], [*tokens], [*split_sizes]


_simple_bert_tokenize_sentences = \
    lambda list_of_texts: [merge_subtokens(BERT_TOKENIZER.tokenize(text)[:BERT_MAX_SEQ_LEN - 2]) for text in
                           list_of_texts]


def bert_tokenize(batch_sentences):
    """
    inputs:
        batch_sentences: List[str]
            a list of textual sentences to tokenized
    outputs:
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    batch_sentences, batch_tokens, batch_splits = _custom_bert_tokenize_sentences(batch_sentences)

    # max_seq_len = max([len(tokens) for tokens in batch_tokens])
    # batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens,max_length=max_seq_len,pad_to_max_length=True) for tokens in batch_tokens]
    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    batch_attention_masks = pad_sequence(
        [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)
    batch_input_ids = pad_sequence([torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts],
                                   batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(
        [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)

    batch_bert_dict = {"attention_mask": batch_attention_masks,
                       "input_ids": batch_input_ids,
                       "token_type_ids": batch_token_type_ids}

    # if len(batch_chunks)>0:
    #     assert sum(batch_chunks)==len(batch_text_pairs)
    #     batch_attention_masks = torch.split(batch_attention_masks,batch_chunks)
    #     batch_input_ids = torch.split(batch_input_ids,batch_chunks)
    #     batch_token_type_ids = torch.split(batch_token_type_ids,batch_chunks)

    # if batchify and len(batch_chunks)>0:
    #     # convert lists obtained in above condition into tensors thorugh zero padding
    #     batch_attention_masks = pad_sequence(batch_attention_masks,batch_first=True,padding_value=0)
    #     batch_input_ids = pad_sequence(batch_input_ids,batch_first=True,padding_value=0)
    #     batch_token_type_ids = pad_sequence(batch_token_type_ids,batch_first=True,padding_value=0)

    # print(batch_attention_masks.shape, batch_input_ids.shape, batch_token_type_ids.shape)

    # return batch_attention_masks, batch_input_ids, batch_token_type_ids, batch_splits
    return batch_sentences, batch_bert_dict, batch_splits


def bert_tokenize_for_valid_examples(batch_orginal_sentences, batch_noisy_sentences):
    """
    inputs:
        batch_noisy_sentences: List[str]
            a list of textual sentences to tokenized
        batch_orginal_sentences: List[str]
            a list of texts to make sure lengths of input and output are same in the seq-modeling task
    outputs (only of batch_noisy_sentences):
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    _batch_orginal_sentences = _simple_bert_tokenize_sentences(batch_orginal_sentences)
    _batch_noisy_sentences, _batch_tokens, _batch_splits = _custom_bert_tokenize_sentences(batch_noisy_sentences)

    valid_idxs = [idx for idx, (a, b) in enumerate(zip(_batch_orginal_sentences, _batch_noisy_sentences)) if
                  len(a.split()) == len(b.split())]
    batch_orginal_sentences = [line for idx, line in enumerate(_batch_orginal_sentences) if idx in valid_idxs]
    batch_noisy_sentences = [line for idx, line in enumerate(_batch_noisy_sentences) if idx in valid_idxs]
    batch_tokens = [line for idx, line in enumerate(_batch_tokens) if idx in valid_idxs]
    batch_splits = [line for idx, line in enumerate(_batch_splits) if idx in valid_idxs]

    batch_bert_dict = {"attention_mask": [], "input_ids": [], "token_type_ids": []}
    if len(valid_idxs) > 0:
        batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]
        batch_attention_masks = pad_sequence(
            [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_input_ids = pad_sequence(
            [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_token_type_ids = pad_sequence(
            [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_bert_dict = {"attention_mask": batch_attention_masks,
                           "input_ids": batch_input_ids,
                           "token_type_ids": batch_token_type_ids}

    return batch_orginal_sentences, batch_noisy_sentences, batch_bert_dict, batch_splits

################################################
# <-----
################################################
