import logging
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

import Utils

class EvaluationMeasures :
    # ---------- Initialization ---------- #
    def __init__(self):
        self.reset_counters()

    def reset_counters(self):
        self.predicted_labels = []
        self.correct_labels = []

        self.total_loss = 0
        self.number_of_steps = 0

    def set_correct_labels(self, labels_ls):
        self.correct_labels = labels_ls

    # ---------- Update ---------- #
    def append_labels(self, labels_t):
        labels_ls = [l.item() for l in list(labels_t)]
        self.predicted_labels = self.predicted_labels + labels_ls

    def append_correct_labels(self, labels_t):
        correct_labels_ls = [l.item() for l in list(labels_t)]
        self.correct_labels = self.correct_labels + correct_labels_ls

    def append_loss(self, loss):
        self.total_loss = self.total_loss + loss
        self.number_of_steps = self.number_of_steps + 1


    # ---------- Evaluation measures ---------- #
    def compute_accuracy(self):
        return accuracy_score(y_true=self.correct_labels, y_pred=self.predicted_labels)

    def compute_precision(self):
        avg_precision = precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        precision_ls = precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return avg_precision, precision_ls

    def compute_recall(self):
        avg_recall = recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        recall_ls = recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return avg_recall, recall_ls

    def compute_f1score(self):
        avg_f1_score =  f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        f1_score_ls = f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return avg_f1_score, f1_score_ls

    def compute_loss(self):
        return self.total_loss / self.number_of_steps

    def compute_confusion_matrix(self):
        return confusion_matrix(self.correct_labels, self.predicted_labels)


def log_accuracy_measures(measures_obj):

    accuracy = measures_obj.compute_accuracy()
    avg_precision, precision_ls = measures_obj.compute_precision()
    avg_recall, recall_ls = measures_obj.compute_recall()
    avg_f1_score, f1_score_ls = measures_obj.compute_f1score()

    loss = measures_obj.compute_loss()

    logging.info("loss=" + str(round(loss,2))+ " ; accuracy=" + str(round(accuracy,3)))
    logging.info("Average F1_score=" + str(round(avg_f1_score, 2)))

