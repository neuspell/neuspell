
SEED = 11690

clean_lines = [line.strip() for line in \
		open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/train.1blm", "r")]
corrupt_lines_prob = [line.strip() for line in \
		open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/train.1blm.noise.prob", "r")]
corrupt_lines_word = [line.strip() for line in \
		open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/train.1blm.noise.word", "r")]

assert len(clean_lines)==len(corrupt_lines_prob)==len(corrupt_lines_word)

import numpy as np
len_ = len(clean_lines)
inds_shuffled = np.arange(len_)
np.random.seed(SEED)
np.random.shuffle(inds_shuffled)

prob_ratio_ = 0.5
prob_len_ = int(np.ceil(prob_ratio_*len_))
prob_indices = inds_shuffled[:prob_len_]
word_indices = inds_shuffled[prob_len_:]

new_clean_lines = []
new_corrupt_lines = []

new_clean_lines += [clean_lines[i] for i in prob_indices]
new_corrupt_lines += [corrupt_lines_prob[i] for i in prob_indices]\

new_clean_lines += [clean_lines[i] for i in word_indices]
new_corrupt_lines += [corrupt_lines_word[i] for i in word_indices]

assert len(new_clean_lines)==len(new_corrupt_lines)

opfile1 = open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/train.1blm.v2", "w")
opfile2 = open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/train.1blm.v2.noise.probword", "w")
for x,y in zip(new_clean_lines, new_corrupt_lines):
	opfile1.write(x+"\n")
	opfile2.write(y+"\n")
opfile1.close()
opfile2.close()


""" # python analyze_dataset.py False train.1blm.v2 train.1blm.v2.noise.probword False
total lines in corrf_lines: 1365669
total tokens in corrf_lines: 30431058
False
total lines in retokenized_lines: 1365669
total tokens in retokenized_lines: 30431058

total lines in incorrf_lines: 1365669
total tokens in incorrf_lines: 30431058
False
total lines in retokenized_lines: 1365669
total tokens in retokenized_lines: 30431058

1365669it [00:11, 123965.81it/s]
number of corr tokens: 24456894, number of incorrect tokens: 5974164, the mistakes %: 19.63
"""
