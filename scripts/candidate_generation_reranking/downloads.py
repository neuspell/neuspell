import os

def _list_files(path: str = os.curdir):
    #print(os.listdir(os.curdir))
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(onlyfiles)
    return onlyfiles

def download_glove(path_to_download: str,
                    size_str: str = "6B"): #"../data/glove/large_files"
    """
    size_str: one of ["6B", "42B.300d", "840B.300d", "twitter.27B"]
    # GloVe dataset
    # to be downloaded from http://nlp.stanford.edu/data/glove.6B.zip"

    """
    # check if directory exists
    if os.path.exists(path_to_download):
        print("specified download directory path already exists. download unsuccessful.")
        return
    os.makedirs(path_to_download)
    os.system("wget -P {} http://nlp.stanford.edu/data/glove.{}.zip".format(path_to_download,size_str))
    os.system("unzip {} -d {}".format(os.path.join(path_to_download,"glove.6B.zip"), path_to_download))
    return


def download_tateoba_english_sentences(path_to_download): #"../data/train/tatoeba/large_files"
    """
    # TATOEBA dataset
    # to be downloaded from http://downloads.tatoeba.org/exports/sentences.tar.bz2
    """
    # check if directory exists
    if os.path.exists(path_to_download):
        print("specified download directory path already exists. download unsuccessful.")
        return
    os.makedirs(path_to_download)
    os.system("wget -P {} http://downloads.tatoeba.org/exports/sentences.tar.bz2".format(path_to_download))
    os.system("tar -C {} -xvf {}".format(path_to_download, os.path.join(path_to_download,"sentences.tar.bz2")))
    lines = open(os.path.join(path_to_download,'sentences.csv')).readlines()
    lines = [i.strip().split('\t') for i in lines][1:]
    sentences = [i[2] for i in lines if i[1] == 'eng']

    write_path = os.path.join(path_to_download,"sentences_eng.txt")
    opfile = open(write_path,"w")
    for sent in set(sentences):
        opfile.write("{}\n".format(sent))
    opfile.close()
    print(write_path)
    return write_path


def download_snli_english_sentences(path_to_download: str): #"../data/train/snli/large_files"
    """
    # SNLI dataset
    # to be downloaded from https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    """
    if os.path.exists(path_to_download):
        print("specified download directory path already exists. download unsuccessful.")
        return
    os.makedirs(path_to_download)
    os.system("wget -P {} https://nlp.stanford.edu/projects/snli/snli_1.0.zip".format(path_to_download))
    os.system("unzip {} -d {}".format(os.path.join(path_to_download,"snli_1.0.zip"), path_to_download))
    sentences = []
    for typ in ["train","dev","test"]:       
        openfile = open(os.path.join(path_to_download, "snli_1.0/snli_1.0_{}.txt".format(typ)), "r")
        for i, line in enumerate(openfile):
          if(i==0):  continue;
          sentences += line.strip().split("\t")[5:7]
        openfile.close()

    write_path = os.path.join(path_to_download,"sentences_eng.txt")
    opfile = open(write_path,"w")
    for sent in set(sentences):
        opfile.write("{}\n".format(sent))
    opfile.close()
    return write_path









    """
def download_hw2_resources(path_to_download):
    if not os.path.exists(path_to_download):
        print("the path_to_download specified as {} doesn't exist".format(path_to_download))
        os.makedirs(path_to_download)
        
    handout_path = os.path.join(path_to_download,"hw2-handout.zip")
    if not os.path.exists(handout_path):
        # -P if specifying a folder name
        # -O if specifying a file name
        # os.system("wget -P {} --user student --password f19textmining \
        #   http://nyc.lti.cs.cmu.edu/classes/11-741/f19/HW/HW2/hw2-handout.zip".format(path_to_download))
        os.system("wget -O {} --user student --password \
            f19textmining http://nyc.lti.cs.cmu.edu/classes/11-741/f19/HW/HW2/hw2-handout.zip".format(handout_path))
        os.system("unzip {} -d {}".format(handout_path, path_to_download))
        
        data_path = os.path.join(path_to_download,"data.tar.gz")
        os.system("tar -C {} -xvf {}".format(path_to_download, data_path))
        os.system("ls")
    return
"""
