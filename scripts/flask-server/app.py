"""
Usage
-----
CUDA_VISIBLE_DEVICES=0 python app.py
"""

import os
from time import time

from flask import Flask, render_template, request
from flask_cors import CORS
from neuspell import BertChecker
from neuspell import BertsclstmChecker
from neuspell import CnnlstmChecker
from neuspell import NestedlstmChecker
from neuspell import SclstmChecker
from neuspell import SclstmbertChecker
from neuspell.seq_modeling.util import is_module_available

# from neuspell import AspellChecker, JamspellChecker

if is_module_available("allennlp"):
    from neuspell import ElmosclstmChecker, SclstmelmoChecker
else:
    msg = "Install `allennlp` by running `pip install -r extras-requirements.txt`. See `README.md` for more info. "
    print("Warning: Not loading ELMO based models. " + msg)

TOKENIZE = True
PRELOADED_MODELS = {}
CURR_MODEL_KEYWORD = "bert"
CURR_MODEL = None
TOPK = 1
LOGS_PATH = "./logs"
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
opfile = open(os.path.join(LOGS_PATH, str(time()) + ".logs.txt"), "w")

# Define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default


@app.route('/')
@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')


@app.route('/loaded', methods=['POST'])
def loaded():
    global CURR_MODEL_KEYWORD, CURR_MODEL
    print(request.form)
    print(request.form["checkers"])
    CURR_MODEL_KEYWORD = request.form["checkers"]
    CURR_MODEL = load_model(CURR_MODEL_KEYWORD)
    return render_template('loaded.html')


@app.route('/reset', methods=['POST'])
def reset():
    return render_template('loaded.html')


@app.route('/predict', methods=['POST'])
def predict():
    global CURR_MODEL, CURR_MODEL_KEYWORD, TOPK
    if request.method == 'POST':
        print("#################")
        print(request.form)
        print(request.form.keys())
        message = request.form['hidden-message']
        message = message.strip("\n").strip("\r")
        if message == "":
            return render_template('loaded.html')
        if TOPK == 1:
            message_modified, result = CURR_MODEL.correct_string(message, return_all=True)
            print(message)
            print(message_modified)
            print(result)
            save_query(CURR_MODEL_KEYWORD + "\t" + message + "\t" + message_modified + "\t" + result + "\n")
            paired = [(a, b) if a == b else ("+-+" + a + "-+-", "+-+" + b + "-+-") for a, b in
                      zip(message_modified.split(), result.split())]
            print(paired)
            return render_template('result.html', prediction=" ".join([x[1] for x in paired]),
                                   message=" ".join([x[0] for x in paired]))
        else:
            raise NotImplementedError("please keep TOPK=1")
        # results = PRELOADED_MODELS[CURR_MODEL_KEYWORD].correct_strings_for_ui([message], topk=TOPK)
        # save_query(CURR_MODEL_KEYWORD+"\t"+message+"\t"+"\t".join(results)+"\n")
        # return render_template('results.html', prediction=results, message=message)	
    return render_template('home.html')


def load_model(model_keyword="bert"):
    global PRELOADED_MODELS
    if model_keyword in PRELOADED_MODELS:
        return PRELOADED_MODELS[model_keyword]

    try:
        if model_keyword == "aspell":
            # return AspellChecker(tokenize=TOKENIZE)
            raise Exception("Not enabled. Install required modules and uncomment this to enable")
        elif model_keyword == "jamspell":
            # return JamspellChecker(tokenize=TOKENIZE)
            raise Exception("Not enabled. Install required modules and uncomment this to enable")
        elif model_keyword == "cnn-rnn":
            return CnnlstmChecker(tokenize=TOKENIZE, pretrained=True)
        elif model_keyword == "sc-rnn":
            return SclstmChecker(tokenize=TOKENIZE, pretrained=True)
        elif model_keyword == "nested-rnn":
            return NestedlstmChecker(tokenize=TOKENIZE, pretrained=True)
        elif model_keyword == "bert":
            return BertChecker(tokenize=TOKENIZE, pretrained=True)
        elif model_keyword == "bertsc-rnn":
            return BertsclstmChecker(tokenize=TOKENIZE, pretrained=True)
        elif model_keyword == "scrnn-bert":
            return SclstmbertChecker(tokenize=TOKENIZE, pretrained=True)
        elif "elmo" in model_keyword:
            try:
                if model_keyword == "elmosc-rnn":
                    return ElmosclstmChecker(tokenize=TOKENIZE, pretrained=True)
                elif model_keyword == "scrnn-elmo":
                    return SclstmelmoChecker(tokenize=TOKENIZE, pretrained=True)
            except ModuleNotFoundError as e:
                msg = "Install `allennlp` by running `pip install -r extras-requirements.txt`. See `README.md` for more info. "
                raise ModuleNotFoundError(msg) from e
        else:
            raise NotImplementedError(f"unknown model_keyword: {model_keyword}")
    except ModuleNotFoundError as e:
        print(e)
    return


def preload_models():
    print("pre-loading models")
    global PRELOADED_MODELS
    PRELOADED_MODELS = {
        # "aspell": AspellChecker(),
        # "jamspell": JamspellChecker(), 
        # "cnn-rnn": CnnlstmChecker(pretrained=True),
        # "sc-rnn": SclstmChecker(tokenize=TOKENIZE, pretrained=True),
        # "nested-rnn": NestedlstmChecker(pretrained=True),
        "bert": BertChecker(tokenize=TOKENIZE, pretrained=True),
        # "elmosc-rnn": ElmosclstmChecker(tokenize=TOKENIZE, pretrained=True),
        # "scrnn-elmo": SclstmelmoChecker(pretrained=True),
        # "bertsc-rnn": BertsclstmChecker(pretrained=True),
        # "scrnn-bert": SclstmbertChecker(pretrained=True)
    }
    print("\n")
    for k, v in PRELOADED_MODELS.items():
        print(f"{k}: {v}")
    print("\n")
    return


def save_query(text):
    global opfile
    opfile.write(text)
    opfile.flush()
    return


if __name__ == "__main__":
    print("*** Flask Server ***")
    preload_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
