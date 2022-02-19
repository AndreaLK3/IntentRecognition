import Filepaths as F
import Model
import Utils
import nltk

# Iterator
def instance_iterator(df):

    class_names = list(df[Utils.INTENT].value_counts().index)
    class_names.sort()

    intent_labels = get_labels(df, class_names)
    utterances_ls = list(df[Utils.UTTERANCE])

    for i, sentence in enumerate(utterances_ls):
        yield (sentence, intent_labels[i])


def get_labels(df, class_names):

    labels_ls = []
    for index, row in df.iterrows():
        labels_ls.append(class_names.index(row[Utils.INTENT]))

    return labels_ls



