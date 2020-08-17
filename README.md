<h2 align="center">
<p>NeuSpell: A Neural Spelling Correction Toolkit
</h3>

### Introduction
NeuSpell is an open-source toolkit for context sensitive spelling correction in English. This toolkit comprises of 10 spell checkers, with evaluations on naturally occurring mis-spellings from multiple (publicly available) sources. To make neural models for spell checking context dependent, (i) we train neural models using spelling errors in context, synthetically constructed by reverse engineering isolated mis-spellings; and  (ii) use richer representations of the context.This toolkit enables NLP practitioners to use our proposed and existing spelling correction systems, both via a simple unified command line,  as well as a web interface. Among many potential applications, we demonstrate the utility of our spell-checkers in combating adversarial misspellings.

##### Demo available at <http://neuspell.github.io/>
![alt text](https://github.com/neuspell/neuspell/blob/master/images/ui.png?raw=true)

##### List of neural models in the toolkit:

- [```CNN-LSTM```](https://drive.google.com/file/d/14XiDY4BJ144fVGE2cfWfwyjnMwBcwhNa/view?usp=sharing)
- [```SC-LSTM```](https://drive.google.com/file/d/1OvbkdBXawnefQF1d-tUrd9lxiAH1ULtr/view?usp=sharing)
- [```Nested-LSTM```](https://drive.google.com/file/d/19ZhWvBaZqrsP5cGqBJdFPtufdyBqQprI/view?usp=sharing)
- [```BERT```](https://huggingface.co/transformers/bertology.html)
- [```SC-LSTM plus ELMO (at input)```](https://drive.google.com/file/d/1mjLFuQ0vWOOpPqTVkFZ_MSHiuVUmgHSK/view?usp=sharing)
- [```SC-LSTM plus ELMO (at output)```](https://drive.google.com/file/d/1P8vX9ByOBQpN9oeho_iOJmFJByv1ifI5/view?usp=sharing)
- [```SC-LSTM plus BERT (at input)```](https://huggingface.co/transformers/bertology.html)
- [```SC-LSTM plus BERT (at output)```](https://huggingface.co/transformers/bertology.html)

![alt text](https://github.com/neuspell/neuspell/blob/master/images/pipeline.jpeg?raw=true)
This pipeline corresponds to the ```SC-LSTM plus ELMO (at input)``` model.

##### Command line usage
You can also find this quick-start code snippet in the ```test_neuspell.py``` file
![alt text](https://github.com/neuspell/neuspell/blob/master/images/cmd.png?raw=true)

### Evaluations
| Spell<br>Checker    | Word<br>Correction <br>Rate | Time per<br>sentence <br>(in milliseconds) |
|----------|----------------------|--------------------------------------|
| ```Aspell``` | 48.7 | 7.3* |
|``` Jamspell``` | 68.9 | 2.6* |
|```CNN-LSTM``` |75.8 |  4.2|
|```SC-LSTM``` | 76.7| 2.8 |
|```Nested-LSTM``` |77.3 | 6.4|
|```BERT``` | 79.1| 7.1|
|```SC-LSTM plus ELMO (at input)``` |<b> 79.8</b>|15.8 |
|```SC-LSTM plus ELMO (at output)``` | 78.5| 16.3|
|```SC-LSTM plus BERT (at input)``` | 77.0| 6.7|
|```SC-LSTM plus BERT (at output)``` | 76.0| 7.2|

Performance of different correctors in the NeuSpell  toolkit  on  the  ```BEA-60K```  dataset  with  real-world  spelling  mistakes. 
∗ indicates  evaluation  on  a CPU (for others we use a GeForce RTX 2080 Ti GPU).


### Using trained models

##### Checkpoints
Run the following to download checkpoints of all neural models
```
cd data/checkpoints
python download_checkpoints.py 
```
See ```data/checkpoints/README.md``` for more details. You can alternatively choose to download only selected models' checkpoints.


### Re-training/Fine-tuning

##### Datasets
Run the following to download datasets
```
cd data/traintest
python download_dataset.py 
```
See ```data/traintest/README.md``` for more details.

##### Synthetic Training Dataset Creation
The toolkit offers 4 kinds of noising strategies to generate synthetic parallel training data to train neural models for spell correction. 
- ```RANDOM```
- ```Word Replacement```
- ```Probabilistic Replacement```
- A combination of ```Word Replacement``` and ```Probabilistic Replacement```
Train files are dubbed with names ```.random```, ```.word```, ```.prob```, ```probword``` respectively. For each strategy, we noise ∼20% of the tokens in the clean corpus. We use 1.6 Million sentences from the [```One billion word benchmark```](https://arxiv.org/abs/1312.3005) dataset as our clean corpus.

### Applications for practitioners
- Defenses against adversarial attacks in NLP
- Improving OCR text correction systems
- Improving grammatical error correction systems
- Improving Intent/Domain classifiers in conversational  AI
- Spell Checking in Collaboration and Productivity tools

### Requirements
The toolkit was developed in python 3.7. 

Required packages can be installed as:
```
pip install -r requirements.txt
```
The code requires ```spacy models``` which can be downloaded as:
```
python -m spacy download en_core_web_sm
```
Requirements for ```Aspell``` checker:
```
wget https://files.pythonhosted.org/packages/53/30/d995126fe8c4800f7a9b31aa0e7e5b2896f5f84db4b7513df746b2a286da/aspell-python-py3-1.15.tar.bz2
tar -C . -xvf aspell-python-py3-1.15.tar.bz2
cd aspell-python-py3-1.15
python setup.py install
```
Requirements for ```Jamspell``` checker:
```
sudo apt-get install -y swig3.0
wget -P ./ https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
tar xf ./en.tar.gz --directory ./
```