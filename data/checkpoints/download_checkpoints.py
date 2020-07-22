
#############################################
# USAGE
# -----
# cd ./data/checkpoints
# python download_checkpoints.py
#############################################

#taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
    else:
        print(f"{path_} already exists")


if __name__=="__main__":

    # cnn-lstm-probwordnoise
    create_paths("./cnn-lstm-probnoise")
    download_file_from_google_drive('', './cnn-lstm-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './cnn-lstm-probwordnoise/vocab.pkl')

    # scrnn-probwordnoise
    create_paths("./scrnn-probwordnoise")
    download_file_from_google_drive('', './scrnn-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './scrnn-probwordnoise/vocab.pkl')

    # lstm-lstm-probwordnoise
    create_paths("./lstm-lstm-probwordnoise")
    download_file_from_google_drive('', './lstm-lstm-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './lstm-lstm-probwordnoise/vocab.pkl')

    # subwordbert-probwordnoise
    create_paths("./subwordbert-probwordnoise")
    download_file_from_google_drive('', './subwordbert-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './subwordbert-probwordnoise/vocab.pkl')

    # elmoscrnn-probwordnoise
    create_paths("./elmoscrnn-probwordnoise")
    download_file_from_google_drive('', './elmoscrnn-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './elmoscrnn-probwordnoise/vocab.pkl')

    # scrnnelmo-probwordnoise
    create_paths("./scrnnelmo-probwordnoise")
    download_file_from_google_drive('', './scrnnelmo-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './scrnnelmo-probwordnoise/vocab.pkl')

    # bertscrnn-probwordnoise
    create_paths("./bertscrnn-probwordnoise")
    download_file_from_google_drive('', './bertscrnn-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './bertscrnn-probwordnoise/vocab.pkl')

    # scrnnbert-probwordnoise
    create_paths("./scrnnbert-probwordnoise")
    download_file_from_google_drive('', './scrnnbert-probwordnoise/model.pth.tar')
    download_file_from_google_drive('', './scrnnbert-probwordnoise/vocab.pkl')