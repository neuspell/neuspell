import os

DEFAULT_DATA_PATH = os.path.join(os.path.split(__file__)[0], "../data")
print(f"data folder is set to `{DEFAULT_DATA_PATH}` script")

DEFAULT_CHECKPOINTS_PATH = os.path.join(DEFAULT_DATA_PATH, "checkpoints")
DEFAULT_TRAINTEST_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "traintest")
DEFAULT_NOISING_RESOURCES_PATH = os.path.join(DEFAULT_DATA_PATH, "noising_resources")

ALLENNLP_ELMO_PRETRAINED_FOLDER = os.path.join(DEFAULT_DATA_PATH, "allennlp_elmo_pretrained")

# if not os.path.exists(DEFAULT_DATA_PATH):
#     os.makedirs(DEFAULT_DATA_PATH)
