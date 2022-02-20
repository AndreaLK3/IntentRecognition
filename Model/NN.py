import logging

import spacy_universal_sentence_encoder
import torch
from torch.nn import Parameter, functional as tfunc


class FFNet(torch.nn.Module):

    def __init__(self, num_classes):

        super(FFNet, self).__init__()
        self.univ_sentence_encoder = spacy_universal_sentence_encoder.load_model('en_use_lg')
        self.vector_size = 512
        self.ff_nn_layers = torch.nn.ModuleList([torch.nn.Linear(self.vector_size, self.vector_size//2),
                                          torch.nn.Linear(self.vector_size//2, num_classes)])


    def forward(self, batch_of_sentences):
        DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        vectors_ls = [doc.vector for doc in self.univ_sentence_encoder.pipe(batch_of_sentences)]
        x_t = torch.tensor(vectors_ls).to(DEVICE)

        intermediate_rep = self.ff_nn_layers[0](x_t)
        logits = self.ff_nn_layers[1](intermediate_rep)

        predictions_classes = tfunc.log_softmax(logits, dim=1)

        return predictions_classes


class BatchEncoder(torch.nn.Module):

    def __init__(self, num_classes):
        super(BatchEncoder,self).__init__()
        self.univ_sentence_encoder = spacy_universal_sentence_encoder.load_model('en_use_lg')


    def forward(self, sentence):
        return self.univ_sentence_encoder(sentence)


