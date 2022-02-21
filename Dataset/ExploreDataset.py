# Questions:
# - Number of instances per class; bar plot. How much is the dataset imbalanced?
# - Average number of words in an instance, per class; bar plot.
# - Color-matrix, expressing the overlap in vocabulary between any 2 classes
import logging

import pandas as pd

import Filepaths
import Utils
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
# import Dataset.GraphicUtils as GraphicUtils

# Visualization: for each intent, on average how many words are in an utterance?
# Meant to explore how some classes may be more "verbose" than others
def words_in_instances(df):

    fig = plt.figure(2)
    intent_names = list(df[Utils.INTENT].value_counts().index)
    avg_words_per_intent = []

    for c_name in intent_names:
        class_instances = df[df[Utils.INTENT] == c_name][Utils.UTTERANCE].to_list()
        num_articles_per_class = len(class_instances)
        total_words_per_class = len(nltk.tokenize.word_tokenize(" ".join(class_instances), language='english'))
        avg_words_per_intent.append(round(total_words_per_class / num_articles_per_class, 1))

    intents_avgwords = list(zip(intent_names, avg_words_per_intent))
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


# Visualization: the number of words in the vocabulary with a given frequency
def vocab_frequencies(vocab_dict):
    # Utils.init_logging("Vocabulary_frequencies.log")

    max_freq = max(vocab_dict.values())
    logging.info("The 15000 utterances of the training dataset contain in total " + str(sum(list(vocab_dict.values()))) + " words")
    logging.info("The vocabulary has " + str(len(vocab_dict.keys())) + " different words")

    frequencies_of_interest = [1,2,3]

    for freq in frequencies_of_interest:
        words_with_freq = list(vocab_dict.values()).count(freq)
        logging.info("There are " +str(words_with_freq) + " words with frequency=" +str(freq))



def exe():
    training_lls, validation_lls, test_lls = Utils.load_dataset()
    df = pd.DataFrame(training_lls, columns=[Utils.UTTERANCE, Utils.INTENT])
    words_in_instances(df)
    vocab = Utils.get_vocabulary(df)
    vocab_frequencies(vocab)
    return vocab



