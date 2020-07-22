
This module injects noise into clean data corpus based on a large collection of misspelled 
and correct word pairs datasets.

-------------------------------------------
-------------------------------------------

FILES:

The following files are required:
(1) stats.py
(2) utils.py

The following file details how to utilize the files above to inject noise:
(1) main.py

A list of tab-seperated homophones pairs can also be provided in this module which will be utilized 
by utils.py while injecting noises into clean data:
(1) homophones.py
	<en-word-1>\t<en-word-homophone-1>
	<en-word-1>\t<en-word-homophone-2>
	...
	<en-word-N>\t<en-word-homophone-K>

A jupyter notebook with sample outputs is available as follows:
(1) colab/noisyfy_experiments_new.ipynb

The file run.py can be used to perform data noisification on any dataset of english sentences by passing arguments.
The step by step guide is provided in main.py; refer that file for more details.

-------------------------------------------
-------------------------------------------

NOTE:

- In main.py, you can choose an option between three datsets
	- TATOEBA, SNLI, 1-billion-word-language-modeling-benchmark
	- The code for downloading the former two datasets is available in main.py
	- The latter being large in size, can be downloaded manually from 
		https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark
- Upon successful execution of the code, the following files/folders are created:
	- file: moe_misspellings_train_ascii_stats_left_context.json
		- a dictionary for replacement counts; from scratch takes about 30-40 mins to obtain on 
			facebook's MOE corpus of ~18M word pairs
	- folder: ./TATOEBA/ or ./SNLI/
		- created if chose to utilize the corresponding datasets

-------------------------------------------
-------------------------------------------

".nosify/large_files" FOLDER details
This folder is not pushed due to its large size

├── large_files
│   ├── moe_misspellings_train.tsv
│   ├── moe_misspellings_train_ascii.tsv
│   ├── moe_misspellings_train_nonascii.tsv
│   ├── moe_misspellings_train_ascii_stats_left_context.json
│   ├── news.en-00001-of-00100
│   ├── news.en-00002-of-00100
│   ├── news.en-00002-of-00100.noise
│   ├── news.en-00002-of-00100.noise.topk
│   ├── news.en-00005-of-00100
│   ├── news.en-00005-of-00100.noise
│   ├── news.en-00005-of-00100.noise.topk
│   ├── news.en-00025-of-00100
│   ├── news.en-00063-of-00100
│   ├── news.en-00063-of-00100.noise
│   ├── news.en-00063-of-00100.noise.topk
│   ├── news.en-00078-of-00100
│   ├── news.en-00078-of-00100.noise.topk
│   ├── news.en-00086-of-00100
│   ├── news.en-00086-of-00100.noise
│   ├── news.en-00086-of-00100.noise.topk
│   ├── news.en-00090-of-00100
│   ├── news.en-00090-of-00100.noise
│   ├── news.en-00099-of-00100
│   ├── news.en-00099-of-00100.noise
│   └── news.en-00099-of-00100.noise.topk

The first file can be obtained from downloading the fb-moe data (https://github.com/facebookresearch/moe)
The next two files can be obtained while exceuting the script in noisify folder (see noisify_experiments_new.ipynb)
The rest of the files can be obtained at https://drive.google.com/drive/folders/1cHOFqPdMaVEbW2wjMfEVmHaHSrgBSaGj?usp=sharing

-------------------------------------------
-------------------------------------------

FUTURE SCOPE:

- Extension to langauges other than English
- Inclusion of more data for obtaining replacement-in-context statistics
- Inclusion of a combination of left and right contexts
- Phoneme replacements using a phoneme-to-character datasets

-------------------------------------------
-------------------------------------------
