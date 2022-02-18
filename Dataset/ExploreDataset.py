# Questions:
# - Number of instances per class; bar plot. How much is the dataset imbalanced?
# - Average number of words in an instance, per class; bar plot.
# - Color-matrix, expressing the overlap in vocabulary between any 2 classes
import pandas as pd

import Filepaths
import Utils
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
# import Dataset.GraphicUtils as GraphicUtils
import json

# Step 0: load the training dataset from train.csv. Columns: "class", "article" (specified in Utils.Column)
#         Get the name and the number of articles for each class.
def initialize(split_name):

    training_split, test_split, validation_split = Utils.load_dataset()

    if split_name == Utils.TRAINING:
        data_lls = training_split
    elif split_name == Utils.VALIDATION:
        data_lls = validation_split
    else:   # test
        data_lls = test_split

    df = pd.DataFrame(data_lls, columns=[Utils.UTTERANCE, Utils.INTENT])
    class_frequencies = list(df[Utils.INTENT].value_counts())
    class_names = list(df[Utils.INTENT].value_counts().index)

    return df, class_frequencies, class_names


# Visualization: for each intent, on average how many words are in an utterance?
# Meant to explore how some classes may be more "verbose" than others
def words_in_instances(df):

    fig = plt.figure(2)
    class_names = list(df[Utils.INTENT].value_counts().index)
    avg_words_per_class = []

    for c_name in class_names:
        class_instances = df[df[Utils.INTENT] == c_name][Utils.UTTERANCE].to_list()
        num_articles_per_class = len(class_instances)
        total_words_per_class = len(nltk.tokenize.word_tokenize(" ".join(class_instances), language='english'))
        avg_words_per_class.append(round(total_words_per_class / num_articles_per_class, 1))

    intents_avgwords = list(zip(class_names, avg_words_per_class))
    intents_avgwords_1 = sorted(intents_avgwords, key=lambda tpl: tpl[1], reverse=True)

    sorted_intents = [tpl[0] for tpl in intents_avgwords_1]
    sorted_avg = [tpl[1] for tpl in intents_avgwords_1]

    bar_obj = plt.bar(x=sorted_intents[0:5] + sorted_intents[-5:],
                      height=sorted_avg[0:5] + sorted_avg[-5:], width=0.6, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Intent')
    plt.ylabel('Words')
    plt.title("Average words per utterance")
    plt.bar_label(bar_obj, labels=sorted_avg[0:5] + sorted_avg[-5:], label_type="center")

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(Filepaths.DATASET_FOLDER, "avg_words_in_intent.png"))


def get_class_vocabularies(df):
    intent_names = list(df[Utils.INTENT].value_counts().index)

    for i_name in intent_names:
        instances = df[df[Utils.INTENT] == i_name][Utils.UTTERANCE].to_list()
        text = len(nltk.tokenize.word_tokenize(" ".join(instances), language='english'))






def vocabulary_unique(class_names, vocabularies_ls):

    fig = plt.figure(4)
    num_classes = len(class_names)
    unique_vocabulary_fraction_ls = []
    for i in range(num_classes):
        vocab_class_i = vocabularies_ls[i]
        cardinality_vocab_class_i = len(vocab_class_i)
        for j in range(num_classes):
            if j != i:
                vocab_class_j = vocabularies_ls[j]
                vocab_class_i = vocab_class_i.difference(vocab_class_j)
        unique_vocabulary_fraction = round(len(vocab_class_i) / cardinality_vocab_class_i,2)
        unique_vocabulary_fraction_ls.append(unique_vocabulary_fraction)
    bar_obj = plt.bar(x=class_names, height=unique_vocabulary_fraction_ls, width=0.6, color="g")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.title("% of vocabulary unique to the class")
    # plt.grid(color='lightgray', linestyle='-', linewidth=0.2, zorder=-1)
    plt.bar_label(bar_obj, labels=unique_vocabulary_fraction_ls, zorder=5)
    # plt.legend(["Training set", "Test set"])
    plt.show()


def exe():
    training_lls, validation_lls, test_lls = Utils.load_dataset()
    df = pd.DataFrame(training_lls, columns=[Utils.UTTERANCE, Utils.INTENT])
    words_in_instances(df)


    vocabulary_unique(class_names, vocabularies_ls)
    plt.show()

