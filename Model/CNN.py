import logging
import torch
from torch.nn import Parameter, functional as tfunc
import string
import Utils

class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.all_chars_ls = list(string.ascii_lowercase + string.digits + string.punctuation + string.whitespace + "’")
        dim_embs = 256
        self.E = Parameter(torch.rand(len(self.all_chars_ls), dim_embs), requires_grad=True)
        dim_embs = self.E.shape[1]
        dim_conv_out = 128
        self.conv1d_k3 = torch.nn.Conv1d(in_channels=dim_embs, out_channels=dim_conv_out, kernel_size=3, padding=2)
        self.conv1d_k6 = torch.nn.Conv1d(in_channels=dim_embs, out_channels=dim_conv_out, kernel_size=6, padding=5)

       # The global maxpooling operation is handled via torch.nn.functional

        self.linear2classes = torch.nn.Linear(dim_conv_out*2, num_classes)


    def forward(self, batch_of_sentences):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        sentence_tensors_ls = []
        for sentence in batch_of_sentences:
            indices = torch.tensor([(self.all_chars_ls).index(c) for c in sentence]).to(CURRENT_DEVICE)
            sentence_tensor = self.E[indices]
            sentence_tensors_ls.append(sentence_tensor)
        Utils.pad_list_of_tensors(sentence_tensors_ls)

        x_t = torch.stack(sentence_tensors_ls).permute(0,2,1)

        conv1_k3 = torch.tanh(self.conv1d_k3(x_t))
        conv1_k6 = torch.tanh(self.conv1d_k6(x_t))

        features_k3 = tfunc.max_pool1d(conv1_k3, kernel_size=conv1_k3.shape[2])
        features_k6 = tfunc.max_pool1d(conv1_k6, kernel_size=conv1_k6.shape[2])

        sentence_rep = torch.cat([features_k3,features_k6], dim=1).squeeze(2)

        logits_classes = self.linear2classes(sentence_rep)
        predictions_classes = tfunc.log_softmax(logits_classes, dim=1)

        return predictions_classes