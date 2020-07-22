

vocab = []

opfile = open("./train.txt","r")
for line in opfile:
    token = line.strip().split(" ")[0]
    vocab.append( token.lower()  )
opfile.close()

for i,line in enumerate(opfile):
    if i!=0:
        token = line.strip().split(",")[1]
        vocab.append( token.lower()  )
opfile.close()

opfile = open("./phonemedataset.txt","w")
for token in vocab[:-1]:
    opfile.write("{}\n".format(token))
opfile.write("{}".format(vocab[-1]))
opfile.close()