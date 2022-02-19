import logging
from math import inf
import torch
import pandas as pd
import Filepaths
import Model.Reader as Reader
import Utils
import os
import torch.nn.functional as tfunc
from datetime import datetime
import Model.EvaluationMeasures as EV
from Model.USE.NN import FFNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup, training loop + validation
def run_train(learning_rate=5e-5, batch_size = 4):

    # initialize log file
    now = datetime.now()
    dt_string = now.strftime("%d%m-%H%M")
    Utils.init_logging("Training_" + "dt" + dt_string + ".log")  #

    # Get training and validation set
    training_split, validation_split, test_split = Utils.load_dataset()
    training_df = pd.DataFrame(training_split, columns=[Utils.UTTERANCE, Utils.INTENT])
    val_df = pd.DataFrame(validation_split, columns=[Utils.UTTERANCE, Utils.INTENT])

    # initialize model
    class_names = list(training_df[Utils.INTENT].value_counts().index)
    class_names.sort()
    logging.info("class_names = " + str(class_names))
    num_classes = len(class_names)
    model = FFNet(num_classes)

    measures_obj = EV.EvaluationMeasures()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_validation_loss = inf  # for early-stopping
    max_epochs = 40
    current_epoch = 1
    num_training_samples = training_df.index.stop

    while current_epoch <= max_epochs:
        logging.info(" ****** Current epoch: " + str(current_epoch) + " ****** ")
        training_iterator = Reader.instance_iterator(training_df)  # the samples' iterator
        sample_num = 0

        # starting operations on one batch
        optimizer.zero_grad()
        batch_elem = 0
        batch_sentences = []
        labels = []
        while batch_elem < batch_size:
            sentence_text, intent_label = training_iterator.__next__()
            batch_sentences.append(sentence_text)
            labels.append(intent_label)
            batch_elem = batch_elem +1
        sample_num = sample_num + batch_size

        label_logprobabilities = model(batch_sentences)
        y_probvalues, y_predicted = label_logprobabilities.sort(dim=1, descending=True)
        y_true = torch.tensor(labels)

        # loss and step
        loss = tfunc.nll_loss(label_logprobabilities, y_true.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # stats
        y_top_predicted = y_predicted.t()[0]
        measures_obj.append_labels(y_top_predicted)
        measures_obj.append_correct_labels(y_true)
        measures_obj.append_loss(loss.item())

        if sample_num % (num_training_samples // 5) == 0:
            logging.info("Training sample: \t " + str(sample_num) + "/ " + str(num_training_samples) + " ...")

        # end of epoch: print stats, and reset them
        # EV.log_accuracy_measures(measures_obj)
        # measures_obj.reset_counters()
        current_epoch = current_epoch + 1

        # examine the validation set
        validation_loss = evaluation(val_df, model)
        logging.info("validation_loss=" + str(round(validation_loss, 3))
                     + " ; best_validation_loss=" + str(round(best_validation_loss, 3)))
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
        else:
            logging.info("Early stopping")
            break  # early stop

    model_fname = "Model_" + "lr" + str(learning_rate) + ".pt"
    torch.save(model, os.path.join(Filepaths.models_folder, Filepaths.saved_models_subfolder, model_fname))
    return model


# Inference only. Used for the validation set, and possibly any test set
def evaluation(corpus_df, model):
    pass
    # model.eval()  # do not train the model now
    # samples_iterator = CorpusReader.next_featuresandlabel_article(corpus_df)
    #
    # validation_measures_obj = EV.EvaluationMeasures()
    # num_samples = corpus_df.index.stop
    # sample_num = 0
    #
    # for article_indices, article_label in samples_iterator:
    #
    #     x_indices_t = torch.tensor(article_indices).to(DEVICE)
    #     y_t = torch.tensor(article_label).to(DEVICE)
    #     label_probabilities = model(x_indices_t)
    #     predicted_label = torch.argmax(label_probabilities)
    #
    #     loss = tfunc.nll_loss(label_probabilities, y_t.unsqueeze(0))
    #
    #     # stats
    #     validation_measures_obj.append_label(predicted_label.item())
    #     validation_measures_obj.append_correct_label(article_label)
    #     validation_measures_obj.append_loss(loss.item())
    #
    #     sample_num = sample_num+1
    #     if sample_num % (num_samples // 5) == 0:
    #         logging.info("Sample: \t " + str(sample_num) + "/ " + str(num_samples) + " ...")
    #
    # # end of epoch: print stats
    # EV.log_accuracy_measures(validation_measures_obj)
    #
    # model.train()  # training can resume
    #
    # validation_loss = validation_measures_obj.compute_loss()
    #
    # return validation_loss