"""
Usage
-----
CUDA_VISIBLE_DEVICES=0 python app.py
"""

from flask import Flask, render_template, url_for, request
from flask_cors import CORS
import os, sys
from time import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from corrector_aspell import CorrectorAspell
from corrector_jamspell import CorrectorJamspell
from corrector_cnnlstm import CorrectorCnnLstm
from corrector_sclstm import CorrectorSCLstm
from corrector_lstmlstm import CorrectorLstmLstm
from corrector_subwordbert import CorrectorSubwordBert
from corrector_elmosclstm import CorrectorElmoSCLstm
from corrector_sclstmelmo import CorrectorSCLstmElmo
from corrector_bertsclstm import CorrectorBertSCLstm
from corrector_sclstmbert import CorrectorSCLstmBert

TOKENIZE = True
PRELOADED_MODELS = {}
CURR_MODEL_KEYWORD = "elmosc-rnn"
CURR_MODEL = None
TOPK = 1
LOGS_PATH = "./logs"
if not os.path.exists(LOGS_PATH):
	os.makedirs(LOGS_PATH)
opfile = open(os.path.join( LOGS_PATH, str(time())+".logs.txt" ), "w")

# Define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default

@app.route('/')
@app.route('/home', methods=['POST'])
def home():
	return render_template('home.html')

@app.route('/loaded',methods=['POST'])
def loaded():
	global CURR_MODEL_KEYWORD, CURR_MODEL
	print(request.form)
	print(request.form["checkers"])
	CURR_MODEL_KEYWORD = request.form["checkers"]
	CURR_MODEL = load_model(CURR_MODEL_KEYWORD)
	return render_template('loaded.html')

@app.route('/reset',methods=['POST'])
def reset():
	return render_template('loaded.html')

@app.route('/predict',methods=['POST'])
def predict():
	global CURR_MODEL, CURR_MODEL_KEYWORD, TOPK
	if request.method == 'POST':
		print("#################")
		print(request.form)
		print(request.form.keys())
		message = request.form['hidden-message']
		message = message.strip("\n").strip("\r")
		if message=="":
			return render_template('loaded.html') 
		if TOPK==1:
			message_modified, result = CURR_MODEL.correct_string(message, return_all=True)
			print(message)
			print(message_modified)
			print(result)
			save_query(CURR_MODEL_KEYWORD+"\t"+message+"\t"+message_modified+"\t"+result+"\n")
			paired = [(a,b) if a==b else ("+-+"+a+"-+-","+-+"+b+"-+-") for a,b in zip(message_modified.split(),result.split())]
			print(paired)
			return render_template('result.html', prediction=" ".join([x[1] for x in paired]), message=" ".join([x[0] for x in paired]))
		else:
			raise NotImplementedError("please keep TOPK=1")
			# results = PRELOADED_MODELS[CURR_MODEL_KEYWORD].correct_strings_for_ui([message], topk=TOPK)
			# save_query(CURR_MODEL_KEYWORD+"\t"+message+"\t"+"\t".join(results)+"\n")
			# return render_template('results.html', prediction=results, message=message)	
	return render_template('home.html')

def load_model(model_keyword="elmosc-rnn"):
	global PRELOADED_MODELS
	if model_keyword in PRELOADED_MODELS:
		return PRELOADED_MODELS[model_keyword]

	if model_keyword=="aspell":
		return CorrectorAspell(tokenize=TOKENIZE)
	elif model_keyword=="jamspell":
		return CorrectorJamspell(tokenize=TOKENIZE)
	elif model_keyword=="cnn-rnn":
		return CorrectorCnnLstm(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="sc-rnn":
		return CorrectorSCLstm(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="nested-rnn":
		return CorrectorLstmLstm(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="bert":
		return CorrectorSubwordBert(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="elmosc-rnn":
		return CorrectorElmoSCLstm(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="scrnn-elmo":
		return CorrectorSCLstmElmo(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="bertsc-rnn":
		return CorrectorBertSCLstm(tokenize=TOKENIZE, pretrained=True)
	elif model_keyword=="scrnn-bert":
		return CorrectorSCLstmBert(tokenize=TOKENIZE, pretrained=True)
	else:
		raise NotImplementedError(f"unknown model_keyword: {model_keyword}")
	return

def preload_models():
	print("pre-loading models")
	global PRELOADED_MODELS
	PRELOADED_MODELS = {
		# "aspell": CorrectorAspell(),
		# "jamspell": CorrectorJamspell(), 
		# "cnn-rnn": CorrectorCnnLstm(pretrained=True),
		"sc-rnn": CorrectorSCLstm(tokenize=TOKENIZE, pretrained=True),
		# "nested-rnn": CorrectorLstmLstm(pretrained=True),
		"bert": CorrectorSubwordBert(tokenize=TOKENIZE, pretrained=True),
		"elmosc-rnn": CorrectorElmoSCLstm(tokenize=TOKENIZE, pretrained=True),
		# "scrnn-elmo": CorrectorSCLstmElmo(pretrained=True),
		# "bertsc-rnn": CorrectorBertSCLstm(pretrained=True),
		# "scrnn-bert": CorrectorSCLstmBert(pretrained=True)
	}
	print("\n")
	for k,v in PRELOADED_MODELS.items():
		print(f"{k}: {v}")
	print("\n")
	return

def save_query(text):
	global opfile
	opfile.write(text)
	opfile.flush()
	return

if __name__=="__main__":
	print("*** Flask Server ***")
	preload_models()
	app.run(debug=True, host='0.0.0.0', port = 5000)
