
###################################
# script ot getsome quick stats about the (corr,incorr) dataset
#
# USAGE
# -----
# python analyze_dataset.py False test.bea4k test.bea4k.noise False
# python analyze_dataset.py False test.bea60k test.bea60k.noise False
# python analyze_dataset.py False test.jfleg test.jfleg.noise False
#
# python analyze_dataset.py False train.1blm train.1blm.noise.random False
# python analyze_dataset.py False train.1blm train.1blm.noise.word False
# python analyze_dataset.py False train.1blm train.1blm.noise.prob False
# python analyze_dataset.py False test.bea60k.ambiguous_natural_v1 test.bea60k.ambiguous_natural_v1.noise False
# python analyze_dataset.py False test.bea60k.ambiguous_natural_v6.2 test.bea60k.ambiguous_natural_v6.2.noise False
# python analyze_dataset.py False test.bea60k.ambiguous_natural_v6.final test.bea60k.ambiguous_natural_v6.final.noise False
###################################


import os, sys
from tqdm import tqdm

try:
	import spacy
	#import en_core_web_sm
	_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
	spacy_tokenizer = lambda corrf: [token.text for token in _spacy_tokenizer(corrf)]
except ModuleNotFoundError:
	os.system("python -m spacy download en_core_web_sm")
	import spacy
	#import en_core_web_sm
	_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
	spacy_tokenizer = lambda corrf: [token.text for token in _spacy_tokenizer(corrf)]

def Retokenize(RE_TOKENIZE: "str: true or false", original_lines):
	print(RE_TOKENIZE)
	if RE_TOKENIZE.lower()=="true":
		retokenized_lines = []
		pbar = tqdm(total=1)
		i, bsz = 0, 5000
		while i>=0:
			lines = " UNIQUE_SPLITTER ".join([line.strip() for line in original_lines[i:i+bsz]])
			tokens = spacy_tokenizer(lines)
			lines = " ".join(tokens).split("UNIQUE_SPLITTER")
			lines = [line.strip() for line in lines]
			retokenized_lines += lines
			i+=bsz
			pbar.update(bsz/len(original_lines))
			if i>len(original_lines): i=-1
		pbar.close()
		assert len(retokenized_lines) == len(original_lines)
	else:
		retokenized_lines = [line.strip() for line in original_lines]
	print("total lines in retokenized_lines: {}".format(len(retokenized_lines)))
	print("total tokens in retokenized_lines: {}".format(sum([len(line.strip().split()) for line in retokenized_lines])))
	return retokenized_lines



if __name__=="__main__":

	RE_TOKENIZE = sys.argv[1] #str true/false
	CLEAN_FILE_PATH_ = sys.argv[2] #str file path
	NOISE_FILE_PATH_ = sys.argv[3] #str file path
	RE_WRITE_FILE = sys.argv[4] #str true/false

	x = "/".join(CLEAN_FILE_PATH_.split("/")[:-1])
	if x!="" and not os.path.exists(x): os.makedirs(x)
	x = "/".join(NOISE_FILE_PATH_.split("/")[:-1])
	if x!="" and not os.path.exists(x): os.makedirs(x)

	opfile = open(CLEAN_FILE_PATH_,"r")
	corrf_lines = [line.strip() for line in opfile]
	opfile.close()
	print("total lines in corrf_lines: {}".format(len(corrf_lines)))
	print("total tokens in corrf_lines: {}".format(sum([len(line.split()) for line in corrf_lines])))
	corrf_lines = Retokenize(RE_TOKENIZE, corrf_lines)
	if RE_WRITE_FILE.lower()=="true":
		opfile = open(CLEAN_FILE_PATH_,"w")
		for line in tqdm(corrf_lines[:-1]):
		    opfile.write(line+"\n")
		opfile.write(corrf_lines[-1])
		opfile.close()

	print("")
	opfile = open(NOISE_FILE_PATH_,"r")
	incorrf_lines = [line.strip() for line in opfile]
	opfile.close()
	print("total lines in incorrf_lines: {}".format(len(incorrf_lines)))
	print("total tokens in incorrf_lines: {}".format(sum([len(line.split()) for line in incorrf_lines])))
	incorrf_lines = Retokenize(RE_TOKENIZE, incorrf_lines)
	if RE_WRITE_FILE.lower()=="true":
		opfile = open(NOISE_FILE_PATH_,"w")
		for line in tqdm(incorrf_lines[:-1]):
		    opfile.write(line+"\n")
		opfile.write(incorrf_lines[-1])
		opfile.close()

	print("")
	corrtokens, incorrtokens = 0, 0
	n_fully_correct_lines = 0
	for corrline, incorrline in tqdm(zip(corrf_lines,incorrf_lines)):
		fully_correct_line = True
		corrline_tokens = corrline.split()
		incorrline_token = incorrline.split()
		try:
			assert len(corrline_tokens)==len(incorrline_token)
		except:
			print(str(len(corrline_tokens))+" "+corrline+"\n"+str(len(incorrline_token))+" "+incorrline+"\n\n")
		for a,b in zip(corrline_tokens,incorrline_token):
			if a==b: 
				corrtokens+=1
			else: 
				fully_correct_line = False
				incorrtokens+=1
		if fully_correct_line:
			n_fully_correct_lines += 1
	print(f"number of corr tokens: {corrtokens}, number of incorrect tokens: {incorrtokens}, the mistakes %: {100*incorrtokens/(incorrtokens+corrtokens):.2f}")
	print(f"# n_fully_correct_lines = {n_fully_correct_lines}/{len(corrf_lines)}")


"""
python analyze_dataset.py False train.1blm train.1blm.noise.random False
==>
# total lines in corrf_lines: 1365669
# total tokens in corrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# total lines in incorrf_lines: 1365669
# total tokens in incorrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# 1365669it [00:10, 133236.15it/s]
# number of corr tokens: 23349537, number of incorrect tokens: 7081521, the mistakes %: 23.27
# # n_fully_correct_lines = 35308/1365669
"""
"""
python analyze_dataset.py False train.1blm train.1blm.noise.word False
==>
# total lines in corrf_lines: 1365669
# total tokens in corrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# total lines in incorrf_lines: 1365669
# total tokens in incorrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# 1365669it [00:11, 123901.11it/s]
# number of corr tokens: 24348043, number of incorrect tokens: 6083015, the mistakes %: 19.99
# # n_fully_correct_lines = 65818/1365669
"""
"""
python analyze_dataset.py False train.1blm train.1blm.noise.prob False
==>
# total lines in corrf_lines: 1365669
# total tokens in corrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# total lines in incorrf_lines: 1365669
# total tokens in incorrf_lines: 30431058
# False
# total lines in retokenized_lines: 1365669
# total tokens in retokenized_lines: 30431058

# 1365669it [00:12, 111174.87it/s]
# number of corr tokens: 24572050, number of incorrect tokens: 5859008, the mistakes %: 19.25
# # n_fully_correct_lines = 62540/1365669
"""