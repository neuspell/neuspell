


clean_lines = [line.strip() for line in \
open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.bea60k.ambiguous_natural_v7","r")]
corrupt_lines = [line.strip() for line in \
open("/projects/ogma2/users/sjayanth/spell-correction/data/traintest/test.bea60k.ambiguous_natural_v7.noise","r")]

assert len(clean_lines)==len(corrupt_lines)

opfile = open("./test.bea60k.ambiguous_natural_v7.beautify","w")

for i,(a,b) in enumerate(zip(clean_lines,corrupt_lines)):
	opfile.write(str(i+1)+"\n")
	opfile.write(b+"\n")
	newline = " ".join([y if x==y else "**"+y+"**" for x,y in zip(a.split(),b.split())])
	opfile.write(newline+"\n")
	newline = " ".join([x if x==y else "**"+x+"**" for x,y in zip(a.split(),b.split())])
	opfile.write(newline+"\n")
opfile.close()
