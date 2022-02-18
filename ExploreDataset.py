# Use matplotlib, seaborn or other Python libraries for visualization to investigate the dataset
# train.csv and test.csv have a stratified split of 10% for testing and the remaining articles for training
# It is opportune to split another 10% fromm train.csv to get a validation set (or to use N-fold cross-validation)

# - Number of articles per class; bar plot. How much is the dataset imbalanced?
# - Average number of words in an article, per class; bar plot.
# - Color-matrix, expressing the overlap in vocabulary between any 2 classes
# - How much of the vocabulary of a class is unique, i.e. non-overlapping? bar plot
import Filepaths
import Utils
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import DatasetGraphics.GraphicUtils as GraphicUtils


# Step 0: load the training dataset from train.csv. Columns: "class", "article" (specified in Utils.Column)
#         Get the name and the number of articles for each class.
def initialize():
    training_df = Utils.load_split(Utils.Split.TRAIN)
    class_frequencies_training = list(training_df["class"].value_counts())
    class_names = list(training_df["class"].value_counts().index)

    #   Lines used in tests during development, since the test set is smaller than the training set,
    #   while less relevant for purposes of exploration
    # test_df = Utils.load_split(Utils.Split.TEST)
    # class_frequencies_test = list(test_df["class"].value_counts())
    # class_frequencies_total = [class_frequencies_training[i] + class_frequencies_test[i]
    #                           for i in range(len(class_frequencies_training))]

    return training_df, class_frequencies_training, class_names

# Visualization 1: how many articles belong to each class? (Bar plot)
# Useful to be aware of how much the dataset is imbalanced
def num_articles(class_frequencies_training, class_names):

    fig = plt.figure(1)
    bar_obj = plt.bar(x=class_names, height=class_frequencies_training, width=0.6, color="b")

    plt.xticks(rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Number of articles')
    plt.title("Articles per class (training set)")
    plt.bar_label(bar_obj, labels=class_frequencies_training)

    GraphicUtils.save_figure(fig, os.path.join(Filepaths.images_folder, 'Number_of_articles.png'))


# Visualization 2: for each class, on average how many words are in an article?
# Meant to explore how some classes may be more "verbose" than others
def words_in_articles(training_df):

    fig = plt.figure(2)
    class_names = list(training_df["class"].value_counts().index)
    avg_words_per_class = []

    for c_name in class_names:
        class_articles = training_df[training_df[Utils.Column.CLASS.value] == c_name][Utils.Column.ARTICLE.value].to_list()
        num_articles_per_class = len(class_articles)
        total_words_per_class = len(nltk.tokenize.word_tokenize(" ".join(class_articles), language='german'))
        avg_words_per_class.append((int(total_words_per_class / num_articles_per_class)))

    bar_obj = plt.bar(x=class_names, height=avg_words_per_class, width=0.6, color="darkgray")
    plt.xticks(rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Words')
    plt.title("Average words per article (training set)")
    plt.bar_label(bar_obj, labels=avg_words_per_class)
    # plt.legend(["Training set", "Test set"])

    GraphicUtils.save_figure(fig, os.path.join(Filepaths.images_folder, 'Avg_words_in_class.png'))


# Visualization 3: The vocabularies of a class may overlap with that of another class. To which extent?
# We expect that the greater the overlap, the more difficult it will be to distinguish between classes
def vocabulary_overlap(class_names, vocabularies_ls):

    fig, ax = plt.subplots()
    num_classes = len(class_names)
    overlap_matrix = np.zeros((num_classes,num_classes))

    for i in range(num_classes):
        words_class_i = vocabularies_ls[i]
        for j in range(num_classes):
            words_class_j = vocabularies_ls[j]
            cardinality_overlap_i_j = len(words_class_i.intersection(words_class_j))
            cardinality_smaller_class = min(len(words_class_i), len(words_class_j))
            overlap_i_j = cardinality_overlap_i_j / cardinality_smaller_class # len(words_class_i)
            if (i>j):
                overlap_matrix[i, j] = overlap_i_j
            #print("|" + str(class_names[i]) + "|= " + str(len(words_class_i)) +
            #      "; |" + str(class_names[j]) + "|= " + str(len(words_class_j))
            #+ "; |overlap|= " + str(cardinality_overlap_i_j) + "; overlap=" + str(round(overlap_i_j,2)))

    ax.set_xticks(list(range(num_classes)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.imshow(overlap_matrix, cmap=GraphicUtils.create_gyr_colormap())
    plt.colorbar()
    plt.title("Vocabulary overlap between classes")
    plt.show()
    GraphicUtils.save_figure(fig, os.path.join(Filepaths.images_folder, 'Vocabulary_overlap.png'))


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
    plt.xticks(rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.title("% of vocabulary unique to the class")
    # plt.grid(color='lightgray', linestyle='-', linewidth=0.2, zorder=-1)
    plt.bar_label(bar_obj, labels=unique_vocabulary_fraction_ls, zorder=5)
    # plt.legend(["Training set", "Test set"])
    plt.show()
    GraphicUtils.save_figure(fig, os.path.join(Filepaths.images_folder, 'Vocabulary_unique.png'))


def all_visualizations():
    training_df, class_frequencies_training, class_names = initialize()

    num_articles(class_frequencies_training, class_names)

    words_in_articles(training_df)

    vocabularies_ls = GraphicUtils.get_class_vocabularies(training_df, class_names)
    vocabulary_overlap(class_names, vocabularies_ls)

    vocabulary_unique(class_names, vocabularies_ls)
    plt.show()

