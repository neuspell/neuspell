# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

import os

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

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
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
        return True
    else:
        print(f"{path_} already exists")
    return False


URL_MAPPINGS_FOR_LARGE_FILES = {
    "cnn-lstm-probwordnoise": {
        "model.pth.tar": "1wEKynHMlBnw2N65jRw8Xox4fsl8BJpmv",
        "vocab.pkl": "13FS6DCsWwrFKEVZl04ELTQulTVzQ0WvP"
    },
    "scrnn-probwordnoise": {
        "model.pth.tar": "1cG0mduVmF7ChR2AOf58XKm0gsVB5d9aC",
        "vocab.pkl": "1M7MH3bL0pvnN5OoIBIxZV-F7G-XXi7qU"
    },
    "lstm-lstm-probwordnoise": {
        "model.pth.tar": "12gbJgYQ30mAVGgyiZlMd2HcGm8SsysLD",
        "vocab.pkl": "12G4AZEpPkESo0iiGaNDQtYXAcpA67Lfh"
    },
    "subwordbert-probwordnoise": {
        "model.pth.tar": "13FnCUPAG-P0-rFIRNewHYXZzwp4HBIjr",
        "vocab.pkl": "11Bo86aI0MxAU1MHpF-eYfAHg3HqiT9me"
    },
    "elmoscrnn-probwordnoise": {
        "model.pth.tar": "14PnNqziPoO0EcL4L5ykGPTmnr0W8I35o",
        "vocab.pkl": "1HnNTutJgE4T-1WrlKjcvXwGzFSg7As98"
    },
    "scrnnelmo-probwordnoise": {
        "model.pth.tar": "1g9Mu144ZlZUbsFTcEPLos6Tjq3-fm2Iv",
        "vocab.pkl": "1tlQDt4Bs_5ICxq6lbdTEbQKSBk9lAiZl"
    },
    "bertscrnn-probwordnoise": {
        "model.pth.tar": "1nMyoXg49_dl_jiXt9bFo8A4Gnd9XdGD2",
        "vocab.pkl": "1IUsAUSyjNgIB9z0H50U656IFHNKO71ws"
    },
    "scrnnbert-probwordnoise": {
        "model.pth.tar": "1vnJZuDVmEfqM92zrakL638PEKY4RsntH",
        "vocab.pkl": "1DwQhYRUxBpGcjsVwfTPLhsFXUt-x00ib"
    }
}

URL_MAPPINGS_FOR_REGULAR_FILES = {
    "cnn-lstm-probwordnoise": {
        "pytorch_model.bin": "1-Bu9C96Vm2yMjhiHdnGk3tzORnmPl8ns",
        "vocab.pkl": "13FS6DCsWwrFKEVZl04ELTQulTVzQ0WvP"
    },
    "scrnn-probwordnoise": {
        "pytorch_model.bin": "1jlHASZzwJ7EaaVdtlM2w8p0Zl2KbPHFe",
        "vocab.pkl": "1M7MH3bL0pvnN5OoIBIxZV-F7G-XXi7qU"
    },
    "lstm-lstm-probwordnoise": {
        "pytorch_model.bin": "1dBYabwsird8TMHtWSzIKQdpdZDx4LH9p",
        "vocab.pkl": "12G4AZEpPkESo0iiGaNDQtYXAcpA67Lfh"
    },
    "subwordbert-probwordnoise": {
        "pytorch_model.bin": "12wPZCl04-00-6YhsX7EE1cEGYZtgGUvk",
        "vocab.pkl": "11Bo86aI0MxAU1MHpF-eYfAHg3HqiT9me"
    },
    "elmoscrnn-probwordnoise": {
        "pytorch_model.bin": "12FPsZePSTeRMCGVnzqZrlPm4NDl6nKWm",
        "vocab.pkl": "1HnNTutJgE4T-1WrlKjcvXwGzFSg7As98"
    },
    "scrnnelmo-probwordnoise": {
        "pytorch_model.bin": "1WlAjzA5ItELMvzj3DeSDiZWEtJW42Ps2",
        "vocab.pkl": "1tlQDt4Bs_5ICxq6lbdTEbQKSBk9lAiZl"
    },
    "bertscrnn-probwordnoise": {
        "pytorch_model.bin": "1c9C8xgapH4drRHG0cEtC57oqPAgxqPE5",
        "vocab.pkl": "1IUsAUSyjNgIB9z0H50U656IFHNKO71ws"
    },
    "scrnnbert-probwordnoise": {
        "pytorch_model.bin": "1zihDsjSOTD38VvqDS3mGeoZkO_rgax6t",
        "vocab.pkl": "1DwQhYRUxBpGcjsVwfTPLhsFXUt-x00ib"
    }
}

CHECKPOINTS_NAMES = [*URL_MAPPINGS_FOR_REGULAR_FILES.keys()]


def download_pretrained_model_large(ckpt_path: str):
    tag = os.path.split(ckpt_path)[-1]
    if tag not in URL_MAPPINGS_FOR_LARGE_FILES:
        raise Exception(
            f"Tried to load an unknown model - {tag}. Available choices are {[*URL_MAPPINGS_FOR_LARGE_FILES.keys()]}")
    details = URL_MAPPINGS_FOR_LARGE_FILES[tag]
    create_paths(ckpt_path)
    model_url = details["model.pth.tar"]
    vocab_url = details["vocab.pkl"]
    print("Pretrained model downloading start (may take few seconds to couple of minutes based on download speed) ...")
    download_file_from_google_drive(vocab_url, os.path.join(ckpt_path, "vocab.pkl"))
    download_file_from_google_drive(model_url, os.path.join(ckpt_path, "model.pth.tar"))
    print("Pretrained model download success")
    return


def _download_pretrained_model(ckpt_path: str):
    tag = os.path.split(ckpt_path)[-1]
    if tag not in URL_MAPPINGS_FOR_REGULAR_FILES:
        raise Exception(
            f"Tried to load an unknown model - {tag}. Available choices are {[*URL_MAPPINGS_FOR_REGULAR_FILES.keys()]}")
    create_paths(ckpt_path)
    details = URL_MAPPINGS_FOR_REGULAR_FILES[tag]

    vocab_path = os.path.join(ckpt_path, "vocab.pkl")
    if os.path.exists(vocab_path):
        print(f"`vocab.pkl` already exists in {ckpt_path}. Continuing with other downloads ...")
    else:
        vocab_url = details["vocab.pkl"]
        download_file_from_google_drive(vocab_url, vocab_path)

    pytorch_model_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
        print(f"`pytorch_model.bin` already exists in {ckpt_path}. Continuing with other downloads ...")
    else:
        print("Pretrained model downloading start "
              "(may take few seconds to couple of minutes based on download speed) ...")
        model_url = details["pytorch_model.bin"]
        download_file_from_google_drive(model_url, pytorch_model_path)
        print("Pretrained model download success")
    return


def _download_all_pretrained_model():
    from ..commons import ARXIV_CHECKPOINTS
    for ckpt_path in ARXIV_CHECKPOINTS.values():
        _download_pretrained_model(ckpt_path)
    return


def download_pretrained_model(ckpt_path: str):
    if ckpt_path == "_all_":
        _download_all_pretrained_model()
    else:
        _download_pretrained_model(ckpt_path)
    return