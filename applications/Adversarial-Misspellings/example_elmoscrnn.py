
##################################
# USAGE
# -----
# CUDA_VISIBLE_DEVICES=3 python example_elmoscrnn.py
##################################

from tqdm import tqdm

import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/.")

from spell_checkers.corrector_elmosclstm import CorrectorElmoSCLstm
ELMOSCRNN_DATA_FOLDER_PATH = "../../../data"


correctorElmoSCLstm = CorrectorElmoSCLstm(DATA_FOLDER_PATH=ELMOSCRNN_DATA_FOLDER_PATH)

noisy = [
		"Hellow papl this is jkadcsnnnn lmnmfefn sdsvdf gibberssh",
		"Hellow Wald",
		"Hellow wald",
		"Hellow Wald is a simple function to strt wth",
		"They fought a deadly wear !!",
		"beciuse he maled the latter , he got an admmision",
		"beciuse he maled the later , he got an admmision",
		"it raned yesterday", "the nt was tight without loose ends",
		"on the contrry there wan a big oportunyty",
		"te contrry side farm house",
		"he contary advice and agreed on the deal",
		"he contry advice and agreed on the deal",
		"tender yet lacerating and darkly funy tble",
		"nicset atcing I have ever witsesed",
		"Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy, it deosn’t mttaer in waht oredr the ltteers in a wrod are, the olny iprmoetnt tihng is taht the frist and lsat ltteer be at the rghit pclae. The rset can be a toatl mses and you can sitll raed it wouthit porbelm. Tihs is bcuseae the huamn mnid deos not raed ervey lteter by istlef, but the wrod as a wlohe.",
		"Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy , it deos n’t mttaer in waht oredr the ltteers in a wrod are , the olny iprmoetnt tihng is taht the frist and lsat ltteer be at the rghit pclae . The rset can be a toatl mses and you can sitll raed it wouthit porbelm . Tihs is bcuseae the huamn mnid deos not raed ervey lteter by istlef , but the wrod as a wlohe ."]
# noisy = noisy*10

# cleaned = correctorElmoSCLstm.correct_strings(noisy)
cleaned = []
for string_ in tqdm(noisy):
	cleaned_string = correctorElmoSCLstm.correct_string(string_)
	cleaned.append(cleaned_string)

for n,c in zip(noisy,cleaned):
	print("########")
	print(n+"\n"+c)
