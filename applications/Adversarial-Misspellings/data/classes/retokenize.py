
from tqdm import tqdm
import spacy
# import en_core_web_sm
_spacy_tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
spacy_tokenizer = lambda inp: [token.text for token in _spacy_tokenizer(inp)]


def spacy_retokenize(inp):

    retokenized_lines = []
    pbar = tqdm(total=1)
    i, bsz = 0, 5000
    while i>=0:
        lines = " UNIQUE_SPLITTER ".join([line.strip() for line in inp[i:i+bsz]])
        tokens = spacy_tokenizer(lines)
        lines = " ".join(tokens).split("UNIQUE_SPLITTER")
        lines = [line.strip() for line in lines]
        retokenized_lines += lines
        i+=bsz
        pbar.update(bsz/len(inp))
        if i>len(inp): i=-1
    pbar.close()
    assert len(retokenized_lines) == len(inp)
    print("total lines in retokenized_lines: {}".format(len(retokenized_lines)))
    print("total tokens in retokenized_lines: {}".format(sum([len(line.strip().split()) for line in retokenized_lines])))

    return retokenized_lines



if __name__=="__main__":

    RE_TOKENIZE = True
    print(RE_TOKENIZE)

    srcfile = "./test.txt"
    lines = [line.strip() for line in open(srcfile,"r")]
    print("n lines: {len(lines)}")

    if RE_TOKENIZE:
        retokenized_lines = spacy_retokenize(lines)
    else:
        retokenized_lines = lines

    opfile = open("./test_retokenized.txt","w")
    for line in retokenized_lines:
        opfile.write(line+"\n")
    opfile.close()