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
        labels_ls = [l.item() for l in list(labels_t)]
        self.correct_labels = self.correct_labels + labels_ls

    def append_loss(self, loss):
        self.total_loss = self.total_loss + loss
        self.number_of_steps = self.number_of_steps + 1


    # ---------- Evaluation measures ---------- #
    def compute_accuracy(self):
        return accuracy_score(y_true=self.correct_labels, y_pred=self.predicted_labels)

    def compute_precision(self):
        avg_precision = precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        precision_ls = precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return min(precision_ls), avg_precision, max(precision_ls)

    def compute_recall(self):
        avg_recall = recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        recall_ls = recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return min(recall_ls), avg_recall, max(recall_ls)

    def compute_f1score(self):
        avg_f1_score =  f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average="macro")
        f1_score_ls = f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)
        return min(f1_score_ls), avg_f1_score, max(f1_score_ls)

    def compute_loss(self):
        return self.total_loss / self.number_of_steps


def log_accuracy_measures(measures_obj):

    accuracy = measures_obj.compute_accuracy()
    min_precision, avg_precision, max_precision = measures_obj.compute_precision()
    min_recall, avg_recall, max_recall = measures_obj.compute_recall()
    min_f1_score, avg_f1_score, max_f1_score = measures_obj.compute_f1score()

    loss = measures_obj.compute_loss()

    logging.info("loss=" + str(round(loss,2))+ " ; accuracy=" + str(round(accuracy,3)))
    logging.info("Precision: min=" + str(round(min_precision, 3))+ " ,avg=" + str(round(avg_precision, 3))+ " ,max=" + str(round(max_precision, 3)))
    logging.info("Recall:min =" + str(round(min_recall, 3))+ " ,avg=" + str(round(avg_recall, 3))+ " ,max=" + str(round(max_recall, 3)))
    logging.info("F1_score:min =" + str(round(min_f1_score, 3)) + " ,avg=" + str(round(avg_f1_score, 3)) + " ,max=" + str(
        round(max_f1_score, 3)))
