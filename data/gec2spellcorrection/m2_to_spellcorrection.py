
###################################
# USAGE
# -----
# python _spellcorr_from_m2.py -corr ./extracted/test.bea60k -incorr ./extracted/test.bea60k.noise -id 0 ./extracted/test.bea.noise.m2
# python _spellcorr_from_m2.py -corr ./extracted/test.jfleg -incorr ./extracted/test.jfleg.noise -id 0 ./extracted/test.jfleg.noise.m2
####################################

'''
Sublime Regex for R:SPELL only
[S][ ].*$\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n\n
[S][ ].*$\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n\n
[S][ ].*$\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n[A][ ]\d+[ ]\d+\|\|\|R:SPELL\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n\n

Sublime Regex for R:MORPH only
[S][ ].*$\n[A][ ]\d+[ ]\d+\|\|\|R:MORPH\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+\n\n

Sublime Regex for #Rc# only
[A][ ]\d+[ ]\d+\|\|\|#Rc#\|\|\|\w+\|\|\|\w+\|\|\|-NONE-\|\|\|\w+
'''


import argparse

# Apply the edits of a single annotator to generate the corrected sentences.
def main(args, incorr_corr_pairs):
	m2 = open(args.m2_file).read().strip().split("\n\n")
	corr_file = open(args.corr, "w")
	incorr_file = open(args.incorr, "w")
	# Do not apply edits with these error types
	skip = {"noop", "UNK", "Um"}
	retain = {"#Rc#","R:SPELL"}
	count = 0
	
	for instance in m2:
		sent_split = instance.split("\n")
		cor_sent = sent_split[0].split()[1:] # Ignore "S "
		mistake_sent = sent_split[0].split()[1:] # Ignore "S "
		edits = sent_split[1:]
		for edit in edits:
			edit = edit.split("|||")
			if edit[1] in skip: continue # Ignore certain edits
			coder = int(edit[-1])
			if coder != args.id: continue # Ignore other coders
			assert (edit[1] in retain)==True
			span = edit[0].split()[1:] # Ignore "A "
			start = int(span[0])
			end = int(span[1])
			# some mistakens can be multiple words conecutively; ex: "burger king" -> "Burger King"
			if (end-start)==1:
				incorr_corr_pairs.append((cor_sent[start],edit[2]))
				cor_sent[start] = edit[2]
			else:
				edits2 = edit[2].split()
				assert len(edits2) == end-start
				for ind in range(start,end):
					incorr_corr_pairs.append((cor_sent[ind],edits2[ind-start]))
					cor_sent[ind] = edits2[ind-start]			
		if len(cor_sent)==len(mistake_sent) and " ".join(cor_sent)!=" ".join(mistake_sent):
			corr_file.write(" ".join(cor_sent)+"\n")
			incorr_file.write(" ".join(mistake_sent)+"\n")
			count += 1
	print(count)
	corr_file.close()
	incorr_file.close()
	return

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser()
	parser.add_argument("m2_file", help="The path to an input m2 file.")
	parser.add_argument("-corr", help="A path to where we save the output corrected text file.", required=True)
	parser.add_argument("-incorr", help="A path to where we save the input incorrect text file.", required=True)
	parser.add_argument("-id", help="The id of the target annotator in the m2 file.", type=int, default=0)	
	args = parser.parse_args()
	
	incorr_corr_pairs = []
	main(args, incorr_corr_pairs)

	opfile = open('./extracted/incorr_corr_pairs.tsv',"w")
	for pair in incorr_corr_pairs:
		opfile.write(pair[0]+"\t"+pair[1]+"\n")
	opfile.close()
