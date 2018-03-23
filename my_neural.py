import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from layers import GaussianNoise, SelfAttention
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


class AttentiveRNN(nn.Module):
    def __init__(self, embeddings, nclasses=3, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            nclasses ():
        """
        super(AttentiveRNN, self).__init__()

        ########################################################
        # Optional Parameters
        ########################################################
        rnn_size = kwargs.get("rnn_size", 100)
        rnn_layers = kwargs.get("rnn_layers", 1)
        bidirectional = kwargs.get("bidirectional", False)
        noise = kwargs.get("noise", 0.)
        dropout_words = kwargs.get("dropout_words", 0.2)
        dropout_rnn = kwargs.get("dropout_rnn", 0.2)
        trainable_emb = kwargs.get("trainable_emb", False)
        ########################################################

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=embeddings.shape[0],
                                      embedding_dim=embeddings.shape[1])
        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        self.init_embeddings(embeddings, trainable_emb)

        # the dropout "layer" for the word embeddings
        self.drop_emb = nn.Dropout(dropout_words)
        # the gaussian noise "layer" for the word embeddings
        self.noise_emb = GaussianNoise(noise)

        # the RNN layer (or layers)
        self.rnn = nn.LSTM(input_size=embeddings.shape[1],
                           hidden_size=rnn_size,
                           num_layers=rnn_layers,
                           bidirectional=bidirectional,
                           dropout=dropout_rnn,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout_rnn)

        if self.rnn.bidirectional:
            rnn_size *= 2

        self.attention = SelfAttention(rnn_size, batch_first=True)

        # the final Linear layer which maps the representation of the sentence,
        # to the classes
        self.linear = nn.Linear(in_features=rnn_size, out_features=nclasses)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(weights, requires_grad=trainable)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, x, lengths):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)
            lengths (): the lengths of each sentence

        Returns: the logits for each class

        """
        embs = self.embedding(x)
        embs = self.noise_emb(embs)
        embs = self.drop_emb(embs)

        # pack the batch
        packed = pack_padded_sequence(embs, list(lengths.data),
                                      batch_first=True)

        out_packed, (h, c) = self.rnn(packed)

        # unpack output - no need if we are going to use only the last outputs
        out_unpacked, _ = pad_packed_sequence(out_packed, batch_first=True)

        # apply dropout to the outputs of the RNN
        out_unpacked = self.drop_rnn(out_unpacked)

        representations, attentions = self.attention(out_unpacked, lengths)

        # project to the classes using a linear layer
        # Important: we do not apply a softmax on the logits, because we use
        # CrossEntropyLoss() as our loss function, which applies the softmax
        # and computes the loss.
        logits = self.linear(representations)
        return logits


class CNNClassifier(nn.Module):

    def __init__(self, embeddings, **kwargs):
        super(CNNClassifier, self).__init__()
        # input
        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        kernel_dim = kwargs.get("kernel_dim", 100)
        kernel_sizes = kwargs.get("kernel_sizes", (3, 4, 5))
        dropout = kwargs.get("dropout", 0.5)
        output_size = kwargs.get("output_size", 3)
        trainable_emb = kwargs.get("trainable_emb", False)
        noise_emb = kwargs.get("noise", 0.2)
        self.drop_emb = nn.Dropout(dropout)
        self.a_e = kwargs.get("aspect_embeddings")
        #  end of inputs.
        self.noise_emb = GaussianNoise(noise_emb)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init_embeddings(embeddings, trainable_emb)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim+15)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def forward(self, inputs, aspects=[]):

        inputs = self.embedding(inputs)  # we have to concatenate the aspect embedding to each of the sentence word.
        inputs = inputs.div(inputs.norm(p=2, dim=1, keepdim=True))
#        inputs = self.noise_emb(inputs)
        inputs = self.drop_emb(inputs)
        to_concate = torch.FloatTensor(inputs.size())
        if aspects:
            aspect_vector = []
            for aspect in aspects:
                aspect_vector.append(self.a_e[aspect].unsqueeze(1).t())
            for index, batch, a_v in zip(list(range(len(to_concate))), to_concate, aspect_vector):
                for row in range(len(batch)):
                    to_concate[index][row] = a_v
            to_concate = nn.Linear(in_features=300, out_features=15)(Variable(to_concate))
            inputs = torch.cat((inputs, to_concate.cuda()), 2)
        inputs = inputs.unsqueeze(1)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        return out

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(weights, requires_grad=trainable)