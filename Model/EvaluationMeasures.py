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
        labels_ls = list(labels_t)
        self.predicted_labels = self.predicted_labels + labels_ls

    def append_correct_labels(self, labels_t):
        labels_ls = list(labels_t)
        self.correct_labels = self.correct_labels + labels_ls

    def append_loss(self, loss):
        self.total_loss = self.total_loss + loss
        self.number_of_steps = self.number_of_steps + 1

    # ---------- Move tensors from CUDA to cpu to compute the scores ---------- #
    def move_tensors_to_cpu_np(self):
        pass


    # ---------- Evaluation measures ---------- #
    def compute_accuracy(self):
        return accuracy_score(y_true=self.correct_labels, y_pred=self.predicted_labels)

    def compute_precision(self):
        return precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_recall(self):
        return recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_f1score(self):
        return f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_loss(self):
        return self.total_loss / self.number_of_steps


def log_accuracy_measures(measures_obj):

    accuracy = measures_obj.compute_accuracy()
    precision = measures_obj.compute_precision()
    recall = measures_obj.compute_recall()
    f1_score = measures_obj.compute_f1score()

    loss = measures_obj.compute_loss()

    logging.info("loss=" + str(round(loss,2))+ " ; accuracy=" + str(round(accuracy,3)))
    logging.info("precision=" + str(Utils.round_list_elems(precision)) + "\nRecall=" + str(Utils.round_list_elems(recall)))
    logging.info("F1_score=" + str(Utils.round_list_elems(f1_score)))