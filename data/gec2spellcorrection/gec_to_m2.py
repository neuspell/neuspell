

#################################################
# README
# ------
# Inputs to this script must be .m2 files
# Outputs will be .m2 with only spell mistakes 
#	retained
# See 'retain' in myfunc() for more detail
# In input files, all types of mistakes can 
#	be present and that too must be present
#	in ERRANT format
#################################################







from collections import OrderedDict 

bea4k_noise_lines = OrderedDict() # {}
opfile = open('./extracted/test.bea4k.noise','r')
for line in opfile:
	line = line.strip()
	if line!="":
		bea4k_noise_lines[line] = 0
opfile.close()
print(f"number of bea4k_noise_lines: {len(bea4k_noise_lines)}")






###################################################
# simple search
###################################################

# out = open('know_punit_patterns.txt', "w")
# n_instances_matched = 0

# m2_file = 'fce.traindevtest.gold.bea19.m2'
# m2 = open(m2_file).read().strip().split("\n\n")
# i = 0
# for raw_sents in m2:
#     i+=1
#     split_sents = raw_sents.split("\n")
#     sent = " ".join(split_sents[0].split()[1:]) # Ignore "S "
#     if sent in bea4k_noise_lines:
#         n_instances_matched+=1
#         bea4k_noise_lines[sent] += 1
#         out.write(raw_sents+"\n\n")
# print("@@",i)
# print(n_instances_matched)

# m2_file = 'lang8.train.auto.bea19.m2'
# m2 = open(m2_file).read().strip().split("\n\n")
# i = 0
# for raw_sents in m2:
#     i+=1
#     split_sents = raw_sents.split("\n")
#     sent = " ".join(split_sents[0].split()[1:]) # Ignore "S "
#     if sent in bea4k_noise_lines:
#         n_instances_matched+=1
#         bea4k_noise_lines[sent] += 1
#         out.write(raw_sents+"\n\n")
# print("@@",i)
# print(n_instances_matched)

# m2_file = 'wilocness.traindev.gold.bea19.m2'
# m2 = open(m2_file).read().strip().split("\n\n")
# i = 0
# for raw_sents in m2:
#     i+=1
#     split_sents = raw_sents.split("\n")
#     sent = " ".join(split_sents[0].split()[1:]) # Ignore "S "
#     if sent in bea4k_noise_lines:
#         n_instances_matched+=1
#         bea4k_noise_lines[sent] += 1
#         out.write(raw_sents+"\n\n")
# print("@@",i)
# print(n_instances_matched)


# for key,value in bea4k_noise_lines.items():
#     if value==0 or value>1:
#         print(f"{value} {key}")

# out.close()









###################################################
# modified search on 
#   1) fce.traindevtest.gold.bea19.m2
###################################################

# annotator_id = 0

# out = open('test.bea4k.noise.m2', "w")
# n_instances_matched, n_instances_present_in_bea4k, unique_lines = 0, 0, OrderedDict() # {}

# m2_file = 'fce.traindevtest.gold.bea19.m2'
# m2 = open(m2_file).read().strip().split("\n\n")
# n_total_data_instances = 0
# skip = {"noop", "UNK", "Um"}
# retain = {"R:SPELL"}

# for instance in m2:
#     n_total_data_instances+=1
#     sent_plus_edits = instance.split("\n")
#     cor_sent = sent_plus_edits[0].split()[1:] # Ignore "S "
#     edits = sent_plus_edits[1:]
#     offset = 0
	
#     # check if any edits are in retain
#     something_to_retain = False
#     for edit in edits:
#         edit = edit.split("|||") #ex: A 12 13|||R:ADJ|||grateful|||REQUIRED|||-NONE-|||0
#         if edit[1] in skip: continue # Ignore certain edits
#         coder = int(edit[-1])
#         if coder != annotator_id: continue # Ignore other coders
#         if edit[1] in retain: # Retain certain edits only     
#             something_to_retain = True
#             break

#     # instance useful only if something can be retained
#     if something_to_retain:
#         retained_edits = []
#         for edit in edits:
#             edit = edit.split("|||") #ex: A 12 13|||R:ADJ|||grateful|||REQUIRED|||-NONE-|||0
#             if edit[1] in skip: continue # Ignore certain edits
#             coder = int(edit[-1])
#             if coder != annotator_id: continue # Ignore other coders
#             if edit[1] in retain: # Retain certain edits only   
#                 retained_edits.append("|||".join(edit))
#                 continue
#             span = edit[0].split()[1:] # Ignore "A "
#             start = int(span[0])
#             end = int(span[1])
#             cor = edit[2].split()
#             cor_sent[start+offset:end+offset] = cor
#             offset = offset-(end-start)+len(cor)
#         write_string = ""
#         write_string = write_string+" ".join(cor_sent)+"\n"
#         write_string = write_string+"\n".join([edit for edit in retained_edits])+"\n\n"
#         out.write(write_string)
#         unique_lines[write_string] = 1
#         n_instances_matched+=1
#         if " ".join(cor_sent) in bea4k_noise_lines: n_instances_present_in_bea4k+=1

# out.close()
# print(n_total_data_instances,n_instances_present_in_bea4k,n_instances_matched,len(unique_lines))









###################################################
# modified search on 
#   1) fce.traindevtest.gold.bea19.m2
#   2) lang8.train.auto.bea19.m2
#   3) wilocness.traindev.gold.bea19.m2
###################################################

from tqdm import tqdm

def myfunc(out, m2_file, annotator_id):

	n_instances_matched, n_instances_present_in_bea4k, unique_lines = 0, 0, OrderedDict() # {}
	m2 = open(m2_file).read().strip().split("\n\n")
	n_total_data_instances = 0
	skip = {"noop", "UNK", "Um"}
	retain = {"#Rc#","R:SPELL"}

	for instance in tqdm(m2):
		n_total_data_instances+=1
		sent_plus_edits = instance.split("\n")
		cor_sent = sent_plus_edits[0].split()[1:] # Ignore "S "
		edits = sent_plus_edits[1:]
		
		# check if any edits are in retain
		something_to_retain = False
		for edit in edits:
			edit = edit.split("|||") #ex: A 12 13|||R:ADJ|||grateful|||REQUIRED|||-NONE-|||0
			if edit[1] in skip: continue # Ignore certain edits
			coder = int(edit[-1])
			if coder != annotator_id: continue # Ignore other coders
			if edit[1] in retain: # Retain certain edits only     
				something_to_retain = True
				break

		"""
		Note
		- due to some type of modifications, the start and end indices of R:SPELL must be changed before retaining
		"""

		# instance useful only if something can be retained
		if something_to_retain:
			offset = 0
			retained_edits = []
			for edit in edits:
				edit = edit.split("|||") #ex: A 12 13|||R:ADJ|||grateful|||REQUIRED|||-NONE-|||0
				if edit[1] in skip: continue # Ignore certain edits
				coder = int(edit[-1])
				if coder != annotator_id: continue # Ignore other coders
				if edit[1] in retain: # Retain certain edits only
					# convert all coders to 0 while saving
					edit[-1] = '0' 
					# modify start and end spans
					temp = edit[0].split()[1:]
					temp[0] = str(offset+int(temp[0]))
					temp[1] = str(offset+int(temp[1]))
					edit[0] = "A"+" "+temp[0]+" "+temp[1]
					retained_edits.append("|||".join(edit))
					continue
				span = edit[0].split()[1:] # Ignore "A "
				start = int(span[0])
				end = int(span[1])
				cor = edit[2].split()
				cor_sent[start+offset:end+offset] = cor
				offset = offset-(end-start)+len(cor)
			write_string = "S "
			write_string = write_string+" ".join(cor_sent)+"\n"
			write_string = write_string+"\n".join([edit for edit in retained_edits])+"\n\n"
			unique_lines[write_string] = 1
			n_instances_matched+=1
			if " ".join(cor_sent) in bea4k_noise_lines: n_instances_present_in_bea4k+=1
	for wr_string in unique_lines.keys():
		out.write(wr_string)
	print(n_total_data_instances,n_instances_matched,n_instances_present_in_bea4k,len(unique_lines))
	return len(unique_lines)

annotator_ids = [0,1,2,3,4,5]

# out = open('./extracted/test.bea.noise.m2', "w")
# m2_files = ['./large_files/bea/lang8.train.auto.bea19.m2',
# 				'./large_files/bea/wilocness.traindev.gold.bea19.m2',
# 				'./large_files/bea/fce.traindevtest.gold.bea19']

# out = open('./extracted/withoutbea4k.test.bea.noise.m2', "w")
# m2_files = ['./large_files/bea/lang8.train.auto.bea19.m2',
# 				'./large_files/bea/wilocness.traindev.gold.bea19.m2'] 

# out = open('./extracted/test.jfleg.noise.m2', "w")
# m2_files = ['./large_files/jfleg/EACL_exp/m2converter/dev.ref.m2',
# 				'./large_files/jfleg/EACL_exp/m2converter/test.ref.m2']

out = open('./extracted/train.bea.noise.m2', "w")
m2_files = ['./large_files/bea/lang8.train.auto.bea19.m2']

import os
for file in m2_files:
	assert os.path.exists(file)==True

cc = 0

for an_id in annotator_ids:
	print("")
	print("###################################")
	print("###################################")
	print(f"annotator_id: {an_id}")
	c = 0
	for m2_file in m2_files:
		_c = myfunc(out, m2_file, an_id)
		print(f"total instances written to out file are (w/ annotator_id:{an_id}, w/ dataset:{m2_file}): {_c}")
		c += _c
	print(f"total instances written to out file are (w/ annotator_id:{an_id}): {c}")
	cc += c

print("")
print("")
print(f"total instances written to out file are: {cc}")
out.close()










###################################################
# segregate the test.bea4k.noise.m2 file into
#   1) test.bea60k
#   2) test.bea60k.noise
###################################################

# from tqdm import tqdm

# def main(m2_file, f1_file, f2_file, an_id):
#     m2 = open(m2_file).read().strip().split("\n\n")

#     # Do not apply edits with these error types
#     skip = {"noop", "UNK", "Um"}
#     count = 0

#     for sent in tqdm(m2):
#         sent = sent.split("\n")
#         cor_sent = sent[0].split()[1:] # Ignore "S "
#         org_sent = sent[0].split()[1:] # Ignore "S "
#         edits = sent[1:]
#         offset = 0
#         for edit in edits:
#             edit = edit.split("|||")
#             if edit[1] in skip: continue # Ignore certain edits
#             coder = int(edit[-1])
#             if coder != an_id: continue # Ignore other coders
#             span = edit[0].split()[1:] # Ignore "A "
#             start = int(span[0])
#             end = int(span[1])
#             cor = edit[2].split()
#             cor_sent[start+offset:end+offset] = cor
#             offset = offset-(end-start)+len(cor)
#         #if len(cor_sent)==len(org_sent) and " ".join(cor_sent)!=" ".join(org_sent):
#         f1_file.write(" ".join(cor_sent)+"\n")
#         f2_file.write(" ".join(org_sent)+"\n")
#         count+=1

#     print(f"lines written to each of {f1_file} and {f2_file} are {count}")
#     return count

# annotator_ids = [0,1,2,3,4,5]
# f1_file = open("test.bea60k","w")
# f2_file = open("test.bea60k.noise","w")
# m2_files = ['test.bea4k.noise.m2']
# cc = 0

# for an_id in annotator_ids:
#     print("")
#     print("###################################")
#     print("###################################")
#     print(f"annotator_id: {an_id}")
#     c = 0
#     for m2_file in m2_files:
#         c += main(m2_file, f1_file, f2_file, an_id)
#     print(f"total lines written to out file are (w/ annotator_id:{an_id}): {c}")
#     cc += c

# f1_file.close()
# f2_file.close()

# print("")
# print("")
# print(f"total lines written to out file are: {cc}")

