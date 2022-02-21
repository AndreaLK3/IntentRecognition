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
from Model.NN import FFNet
from Model.CNN import ConvNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup, training loop + validation
def run_train(learning_rate=1e-4, batch_size = 8, model_type="FFNet"):

    # initialize log file
    now = datetime.now()
    dt_string = now.strftime("%d%m-%H%M")
    Utils.init_logging("Training_" + "dt" + dt_string + ".log")
    logging.info("Starting training...")
    logging.info("batch_size=" + str(batch_size) + ", learning_rate="+ str(learning_rate))

    # Get training and validation set
    training_split, validation_split, test_split = Utils.load_dataset()
    training_df = pd.DataFrame(training_split, columns=[Utils.UTTERANCE, Utils.INTENT])
    val_df = pd.DataFrame(validation_split, columns=[Utils.UTTERANCE, Utils.INTENT])

    # initialize model
    class_names = list(training_df[Utils.INTENT].value_counts().index)
    class_names.sort()
    logging.info("class_names = " + str(class_names))
    num_classes = len(class_names)
    if model_type == "FFNet":
        model = FFNet(num_classes)
    elif model_type == "ConvNet":
        model = ConvNet(num_classes)
    model.to(DEVICE)

    measures_obj = EV.EvaluationMeasures()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_validation_loss = inf  # for early-stopping
    max_epochs = 100
    current_epoch = 0
    num_training_samples = training_df.index.stop

    while current_epoch < max_epochs:
        current_epoch = current_epoch + 1
        logging.info(" ****** Current epoch: " + str(current_epoch) + " ****** ")
        training_iterator = Reader.instance_iterator(training_df)  # the samples' iterator
        sample_num = 0

        while sample_num < num_training_samples:

            # starting operations on one batch
            optimizer.zero_grad()
            batch_elem = 0
            batch_sentences = []
            labels = []
            while batch_elem < batch_size and sample_num < num_training_samples:
                batch_elem = batch_elem + 1
                sentence_text, intent_label = training_iterator.__next__()
                batch_sentences.append(sentence_text)
                labels.append(intent_label)
                sample_num = sample_num + 1

            label_logprobabilities = model(batch_sentences)
            y_probvalues, y_predicted = label_logprobabilities.sort(dim=1, descending=True)
            y_true = torch.tensor(labels).to(DEVICE)

            # loss and step
            loss = tfunc.nll_loss(label_logprobabilities, y_true)
            loss.backward()
            optimizer.step()

            # stats
            y_top_predicted = y_predicted.t()[0]
            measures_obj.append_labels(y_top_predicted)
            measures_obj.append_correct_labels(y_true)
            measures_obj.append_loss(loss.item())

            if sample_num % (num_training_samples // 10) == 0:
                logging.info("Training sample: \t " + str(sample_num) + "/ " + str(num_training_samples) + " ...")

        # end of epoch: print stats, and reset them
        EV.log_accuracy_measures(measures_obj)
        measures_obj.reset_counters()
        #
        # examine the validation set
        validation_loss = evaluation(val_df, model)
        logging.info("validation_loss=" + str(round(validation_loss, 3))
                     + " ; best_validation_loss=" + str(round(best_validation_loss, 3)))
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
        else:
            logging.info("Early stopping")
            break  # early stop

    model_fname = "Model_" + model.__class__.__name__ + "_dt" + dt_string +  ".pt"
    torch.save(model, os.path.join(Filepaths.MODELS_FOLDER, Filepaths.SAVED_MODELS_SUBFOLDER, model_fname))
    return model


# Inference only. Used for the validation set, and possibly any test set
def evaluation(corpus_df, model):
    logging.info("Evaluation")
    model.eval()  # do not train the model now
    samples_iterator = Reader.instance_iterator(corpus_df)

    measures_obj = EV.EvaluationMeasures()
    num_samples = corpus_df.index.stop
    sample_num = 0
    batch_size = 1

    while sample_num < num_samples:

        batch_elem = 0
        batch_sentences = []
        labels = []
        while batch_elem < batch_size:
            batch_elem = batch_elem + 1
            sentence_text, intent_label = samples_iterator.__next__()
            batch_sentences.append(sentence_text)
            labels.append(intent_label)

        sample_num = sample_num + batch_size

        label_logprobabilities = model(batch_sentences)
        y_probvalues, y_predicted = label_logprobabilities.sort(dim=1, descending=True)
        y_true = torch.tensor(labels).to(DEVICE)

        # loss and step
        loss = tfunc.nll_loss(label_logprobabilities, y_true)

        # stats
        y_top_predicted = y_predicted.t()[0]
        measures_obj.append_labels(y_top_predicted)
        measures_obj.append_correct_labels(y_true)
        measures_obj.append_loss(loss.item())

        sample_num = sample_num+1
        if sample_num % (num_samples // 5) == 0:
            logging.info("Sample: \t " + str(sample_num) + "/ " + str(num_samples) + " ...")

    # end of epoch: print stats
    EV.log_accuracy_measures(measures_obj)
    model.train()  # training can resume
    evaluation_loss = measures_obj.compute_loss()

    return evaluation_loss