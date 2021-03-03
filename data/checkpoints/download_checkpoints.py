#############################################
# USAGE
# -----
# cd ./data/checkpoints
# python download_checkpoints.py
#############################################

# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

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
        return True
    else:
        print(f"{path_} already exists")
    return False


if __name__ == "__main__":

    # cnn-lstm-probwordnoise
    if create_paths("./cnn-lstm-probwordnoise"):
        download_file_from_google_drive('1wEKynHMlBnw2N65jRw8Xox4fsl8BJpmv', './cnn-lstm-probwordnoise/model.pth.tar')
        download_file_from_google_drive('13FS6DCsWwrFKEVZl04ELTQulTVzQ0WvP', './cnn-lstm-probwordnoise/vocab.pkl')

    # scrnn-probwordnoise
    if create_paths("./scrnn-probwordnoise"):
        download_file_from_google_drive('1cG0mduVmF7ChR2AOf58XKm0gsVB5d9aC', './scrnn-probwordnoise/model.pth.tar')
        download_file_from_google_drive('1M7MH3bL0pvnN5OoIBIxZV-F7G-XXi7qU', './scrnn-probwordnoise/vocab.pkl')

    # lstm-lstm-probwordnoise
    if create_paths("./lstm-lstm-probwordnoise"):
        download_file_from_google_drive('12gbJgYQ30mAVGgyiZlMd2HcGm8SsysLD', './lstm-lstm-probwordnoise/model.pth.tar')
        download_file_from_google_drive('12G4AZEpPkESo0iiGaNDQtYXAcpA67Lfh', './lstm-lstm-probwordnoise/vocab.pkl')

    # subwordbert-probwordnoise
    if create_paths("./subwordbert-probwordnoise"):
        download_file_from_google_drive('13FnCUPAG-P0-rFIRNewHYXZzwp4HBIjr', './subwordbert-probwordnoise/model.pth.tar')
        download_file_from_google_drive('11Bo86aI0MxAU1MHpF-eYfAHg3HqiT9me', './subwordbert-probwordnoise/vocab.pkl')

    # elmoscrnn-probwordnoise
    if create_paths("./elmoscrnn-probwordnoise"):
        download_file_from_google_drive('14PnNqziPoO0EcL4L5ykGPTmnr0W8I35o', './elmoscrnn-probwordnoise/model.pth.tar')
        download_file_from_google_drive('1HnNTutJgE4T-1WrlKjcvXwGzFSg7As98', './elmoscrnn-probwordnoise/vocab.pkl')

    # scrnnelmo-probwordnoise
    if create_paths("./scrnnelmo-probwordnoise"):
        download_file_from_google_drive('1g9Mu144ZlZUbsFTcEPLos6Tjq3-fm2Iv', './scrnnelmo-probwordnoise/model.pth.tar')
        download_file_from_google_drive('1tlQDt4Bs_5ICxq6lbdTEbQKSBk9lAiZl', './scrnnelmo-probwordnoise/vocab.pkl')

    # bertscrnn-probwordnoise
    if create_paths("./bertscrnn-probwordnoise"):
        download_file_from_google_drive('1nMyoXg49_dl_jiXt9bFo8A4Gnd9XdGD2', './bertscrnn-probwordnoise/model.pth.tar')
        download_file_from_google_drive('1IUsAUSyjNgIB9z0H50U656IFHNKO71ws', './bertscrnn-probwordnoise/vocab.pkl')

    # scrnnbert-probwordnoise
    if create_paths("./scrnnbert-probwordnoise"):
        download_file_from_google_drive('1vnJZuDVmEfqM92zrakL638PEKY4RsntH', './scrnnbert-probwordnoise/model.pth.tar')
        download_file_from_google_drive('1DwQhYRUxBpGcjsVwfTPLhsFXUt-x00ib', './scrnnbert-probwordnoise/vocab.pkl')
