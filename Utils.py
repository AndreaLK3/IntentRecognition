import Filepaths
import json
import nltk
import torch
import os
import sys
import logging
import torch.nn.functional as tfunc

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

# Given a list of tensors (seql_len, dim_features), pad them all to the same size with zeros
def pad_list_of_tensors(x_t_ls):
    # determine the maximum sequence length
    max_seq_len = 0
    for i in range(len(x_t_ls)):
        seq_len = x_t_ls[i].shape[0]
        max_seq_len = max(seq_len, max_seq_len)

    # pad all sequences to the same length with zeros
    for i in range(len(x_t_ls)):
        seq_len = x_t_ls[i].shape[0]
        # to pad only the last dimension of the input tensor, the pad has the form (padding_left, padding_right)
        x_t_ls[i] = tfunc.pad(x_t_ls[i], (0, 0, 0, max_seq_len - seq_len))


def load_model_from_file(filename):
    saved_model_path = os.path.join(Filepaths.MODELS_FOLDER, Filepaths.SAVED_MODELS_SUBFOLDER, filename)
    model = torch.load(saved_model_path) if torch.cuda.is_available() \
        else torch.load(saved_model_path, map_location=torch.device('cpu'))
    logging.info("Loading the model found at: " + str(saved_model_path))

    return model