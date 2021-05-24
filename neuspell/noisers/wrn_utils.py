import random

from tqdm.autonotebook import tqdm

SEED = 11690


def _load_assorted_mistakes(file_path):
    """load assorted mistakes data; a dict of vocab along with the number of possible replacement candidates
    """
    opfile = open(file_path, "r")
    mistakes_vocab = {}
    for i, line in enumerate(opfile):
        if (i != 0):
            try:
                word, count = line.strip().split("\t")
                mistakes_vocab[word] = count
            except:
                pass
    opfile.close()
    # print(mistakes_vocab)
    return mistakes_vocab


def _load_assorted_mistakes_mappings(file_path):
    """load mistakes mappings; a dict of vocab along with their possible replacement candidates
    """
    mistakes_mappings = {}
    opfile = open(file_path, "r")
    for line in opfile:
        if line:
            error, correction = line.strip().split("\t")
            try:
                mistakes_mappings[correction].append(error)
            except:
                mistakes_mappings[correction] = [error]
    opfile.close()
    return mistakes_mappings


def _calculate_mistaketoken_overlap(original_sentences, mistakes_vocab, return_mode=False):
    """
    find overlap
    to check how many tokens (space seperated) in original_sentences
      match the word-tokens in the misspellings vocab loaded above
    """
    overlap_words, overlap_count, total_count = {}, 0, 0
    for line in tqdm(original_sentences):
        words = line.strip().split()
        for word in words:
            total_count += 1
            if word in mistakes_vocab:
                try:
                    overlap_words[word] += 1
                except:
                    overlap_words[word] = 1
    overlap_count = sum([*overlap_words.values()])
    overlap_percent = 100 * overlap_count / total_count
    print(f"unique tokens overlapped with replacement lookup: {len(overlap_words)}")
    print(f"total tokens overlapped with replacement lookup: {overlap_count}")
    print("overlap percent wrt original_sentences: {:.4f}".format(overlap_percent))
    print("overlap percent wrt mistakes_vocab: {:.4f}".format(100 * len(overlap_words) / len(mistakes_vocab)))
    if return_mode:
        return overlap_words, overlap_count, total_count, overlap_percent
    return


def noisyfy_word_tokens(original_sentences,
                        mistakes_vocab,
                        mistakes_mappings,
                        expected_prob,
                        print_stats=True,
                        min_len=1):
    """
    inject replacements from mistakes_vocab
    expected_prob is the prob of mistakeful tokens you want in your dataset
      after running the noise injection step; firstly the token overlap_percentage
      is computed as only overlapped tokens can be replaced. Then chnace of replacing
      a overlapped token is calculated using expected_prob & overlap_percentage
    """
    assert 0.0 < expected_prob < 1.0
    print("total lines in inp to noisyfy_word_tokens: {}".format(len(original_sentences)))
    print("total tokens in inp to noisyfy_word_tokens: {}".format(
        sum([len(line.strip().split()) for line in original_sentences])))
    # print("------------------------------------")
    overlap_words, overlap_count, total_count, overlap_percent = \
        _calculate_mistaketoken_overlap(original_sentences,
                                        mistakes_vocab,
                                        return_mode=True)
    # print(f"#overlap_count:{overlap_count}, #total_count:{total_count}, #overlap_percent:{overlap_percent}")
    # print("------------------------------------")
    # prob can be calclulated as prob*overlap = 15% as a standard
    prob = expected_prob / (overlap_percent / 100)
    print("{:.4f}% of overlapped tokens will get replaced to "
          "match the total % of misspellings to {:.4f}%".format(100 * prob, 100 * expected_prob))

    get_noisy_token = lambda token: random.choice(mistakes_mappings[token]) \
        if (token in mistakes_vocab and random.uniform(0, 1) <= prob and len(token) > min_len) \
        else token
    error_original_pair = [
        (" ".join([get_noisy_token(token) for token in line.split()]),
         " ".join([token for token in line.split()])) \
        for line in original_sentences
    ]

    if print_stats:
        # print some examples error sentences
        """
        for i, pair in enumerate(error_original_pair):
            if(i>=15): break 
            print(pair[0]+"\n"+pair[1])
        """
        # how many tokens did we replace??
        difflen = [sum([1 if a != b else 0 for a, b in zip(error.split(), original.split())]) \
                   for (error, original) in error_original_pair]
        originallen = [len(original.split()) for (_, original) in error_original_pair]
        print(f"Percentage of tokens that actually got replaced "
              f"{sum(difflen)}/{sum(originallen)}={100 * sum(difflen) / sum(originallen):.4f}%")
        # how many unique tokens did we replace?
        diffvocabdict = {}
        for (error, original) in error_original_pair:
            for a, b in zip(error.split(), original.split()):
                if a != b:
                    try:
                        diffvocabdict[b] += 1
                    except:
                        diffvocabdict[b] = 1
        print("No of tokens in mistakes_mappings queried: {}". \
              format(len(set([*mistakes_mappings.keys()]).intersection(set([*diffvocabdict.keys()])))))
    return [pair[0] for pair in error_original_pair]
