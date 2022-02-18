import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import CountVectorizer

from Utils import Column


def create_gyr_colormap():
    viridisBig = cm.get_cmap('hsv', 1024)
    newcmp = ListedColormap(viridisBig(np.linspace(0.38, 0.08, 1024)))

    return newcmp


def save_figure(fig, out_fpath):
    fig.tight_layout()
    fig.savefig(out_fpath)

# Auxiliary function: use CountVectorizer's tokenization to get the vocabulary (words, no frequencies) of each class
def get_class_vocabularies(training_df, class_names):

    classes_words_ls = []
    for c_name in class_names:
        class_articles = training_df[training_df[Column.CLASS.value] == c_name][
            Column.ARTICLE.value].to_list()
        vectorizer = CountVectorizer(lowercase=True)
        vectorizer.fit(class_articles)
        words_in_class = set(vectorizer.vocabulary_.keys())
        classes_words_ls.append(words_in_class)
    return classes_words_ls