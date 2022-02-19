import Filepaths
import json
import nltk
import torch
import os
import sys
import logging

UTTERANCE = "utterance"
INTENT = "intent"

TRAINING = "training"
VALIDATION = "validation"
TEST = "test"

def load_dataset():
    json_file = open(Filepaths.DATASET, "r")

    json_data = json.load(json_file)

    training_split = json_data["train"]
    validation_split = json_data["val"]
    test_split = json_data["test"]

    return training_split, validation_split, test_split


def get_vocabulary(df):

    vocab_dict = dict()
    utterances = df[UTTERANCE].to_list()

    for sentence in utterances:
        tokens = nltk.tokenize.word_tokenize(sentence, language="english")
        for tok in tokens:
            try:
                vocab_dict[tok] = vocab_dict[tok] + 1
            except KeyError:
                vocab_dict[tok] = 1
    return vocab_dict


# Invoked to write a message to a text logfile and also print it
def init_logging(logfilename, loglevel=logging.INFO):
  for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
  logging.basicConfig(level=loglevel, filename=logfilename, filemode="w",
                      format='%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

  if len(logging.getLogger().handlers) < 2:
      outlog_h = logging.StreamHandler(sys.stdout)
      outlog_h.setLevel(loglevel)
      logging.getLogger().addHandler(outlog_h)


# Round the numbers in a list
def round_list_elems(ls, precision=2):
    rounded_ls = [round(elem, precision) for elem in ls]
    return rounded_ls


def load_model(lr):
    model_fname = "Model_" + "lr" + str(lr) + ".pt"
    saved_model_fpath = os.path.join(Filepaths.MODELS_FOLDER, Filepaths.SAVED_MODELS_SUBFOLDER, model_fname)
    model = torch.load(saved_model_fpath)
    return model