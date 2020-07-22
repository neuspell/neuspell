
#############################################
# USAGE
# -----
# cd ./data/traintest
# python download_datafiles.py
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

    """
    All files from https://drive.google.com/drive/folders/1ejKSkiHNOlupxXVDMg67rPdqwowsTq1i?usp=sharing
    will be downloaded to the current folder directory
    """

    # ./

    download_file_from_google_drive('1ZlEQKf3HMMk66F7DGFPnh-PA2cbt5K0F', 'test.1blm')
    download_file_from_google_drive('1wZ6nrIYANNN3ZoHgacIg9P3UmHnBb9Wa', 'test.1blm.noise.prob')
    download_file_from_google_drive('1epwQQjmOZyZL1ptc9mcIFjnwS0vs7L46', 'test.1blm.noise.random')
    download_file_from_google_drive('1aT3mUfsNtTl51vc-V7kJZeflxZ4BMicD', 'test.1blm.noise.word')

    download_file_from_google_drive('1QxVnFgp0pgEWmS-113SWEjT8tEhXCVF5', 'test.bea4k')
    download_file_from_google_drive('1pnCU3OUSE0lNN1T6qY4WWhtHZsW3cg1c', 'test.bea4k.noise')

    download_file_from_google_drive('1eXrAPKzfU7E9EZNKMyyanuxL9NMpkvdv', 'test.bea20k')
    download_file_from_google_drive('178AWu05IzYFBOFYQ0lhkkBQaIACSJzAC', 'test.bea20k.noise')

    download_file_from_google_drive('10VtrEThrDIiuFJf0gj4LeGDdP-y-yR--', 'test.bea60k')
    download_file_from_google_drive('16AMIb6FVltgRR8xv8h7qacDUX8cOQK9d', 'test.bea60k.noise')

    download_file_from_google_drive('192g_5oJn4dro5QJ88Dd8-lN_xKE_lLf0', 'test.bea322')
    download_file_from_google_drive('1_hka2FOT4FrMvsV3d4Zfi9W8v3oFBRYc', 'test.bea322.noise')

    download_file_from_google_drive('1v0tRcNZctvVGqrmjlda_6dH8AfZUzFGO', 'test.bea4660')
    download_file_from_google_drive('1EmuKeNgBRzc760R0dSSuZ36xdikQRlIS', 'test.bea4660.noise')

    download_file_from_google_drive('1jHR2f3JwnskDphQcaTXr0hLlp60qJxUl', 'test.jfleg')
    download_file_from_google_drive('1sccH7dRhyctKAIQXBZEBmUWEiTN_-o6q', 'test.jfleg.noise')

    download_file_from_google_drive('1aWHIxu_BrZIeGRLhID3J_od6shXz3jUb', 'train.1blm')
    download_file_from_google_drive('16RYImD2esgGwc1nNt3Yf-WR5TU1yQyik', 'train.1blm.noise.prob')
    download_file_from_google_drive('11FMI2C-ouwaWesTLjfPCXmqeB6HUQHkK', 'train.1blm.noise.random')
    download_file_from_google_drive('1eRpWqSb7sIm3kgtkdfVTru9YKHSRRrdq', 'train.1blm.noise.word')
    
    download_file_from_google_drive('1INTWXWO6i1Swthu5ln7REZjGvPFv2hyQ', 'train.bea40k')
    download_file_from_google_drive('1KTeL8oZ30fVI_QuCW879T-CgfeewvSk9', 'train.bea40k.noise')

    download_file_from_google_drive('1s6CQ6NlsstCLbLCSEZyP-uvNwx4UwzZT', 'train.moviereviews')
    download_file_from_google_drive('1xk3jyTkiVEWDsl-Abhc8XUXiIVp_WXCg', 'valid.moviereviews')


    # wo_context
    create_paths("./wo_context")

    download_file_from_google_drive('1uNHQovF2Z0QPp27i2Sq5abVL_f56iNYK', './wo_context/aspell_big')
    download_file_from_google_drive('19eSfVnX-sIdUnaazEaUCsHsFt8p5qoXZ', './wo_context/aspell_big.noise')
    
    download_file_from_google_drive('1XqHN1VnVnVnSR4-wF_iI6VUVPf9S_TP3', './wo_context/aspell_small')
    download_file_from_google_drive('1I9jhthL6y52h8uRuwcnRjhtRbNw3mX8V', './wo_context/aspell_small.noise')
    
    download_file_from_google_drive('1ptBfh8UvbAUH7K1TIALDZM8CUL_Yhxty', './wo_context/combined_data')
    download_file_from_google_drive('1bSye_TITRUdO4CUIt9i1R2PkCdQD346h', './wo_context/combined_data.noise')
    
    download_file_from_google_drive('1bj9zQqntrVRydBn-YHZ0BcT55Xf37jq-', './wo_context/homophones')
    download_file_from_google_drive('1rL_OdqQgr-kL6X_N94epxzpnIzj_L6y6', './wo_context/homophones.noise')


