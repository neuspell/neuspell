
# 1blm_100K_vocab
#
# import pickle
# path_ = "../../../../data/checkpoints/elmoscrnn-probwordnoise/vocab.pkl"
# fp = open(path_,'rb'); vocab = pickle.load(fp);
# v = [*vocab["token2idx"].keys()]; print(len(v))
# opfile = open("1blm_100K_vocab.txt","w")
# for line in v: 
# 	opfile.write(line+"\n")
# opfile.close()

# gpt2_vocab
#
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# vocab = [k if not k.startswith("Ġ") else k[1:] for k in tokenizer.get_vocab().keys()]
# print(vocab)
# opfile = open("gpt2_vocab.txt","w")
# for line in vocab:
#     opfile.write(line+"\n")
# opfile.close()

# common vocab
#
gpt2_vocab = {line.strip():"" for line in open("gpt2_vocab.txt","r")}
1blm_100K_vocab = {line.strip():"" for line in open("1blm_100K_vocab.txt","r")}
intersect_vocab = [word for word in 1blm_100K_vocab if word in gpt2_vocab]
non_intersect_vocab = [word for word in 1blm_100K_vocab if word not in gpt2_vocab]
print(f"# common vocab between gpt2_vocab and 1blm_100K_vocab: {len(intersect_vocab)}")

