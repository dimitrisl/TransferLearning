import xml.etree.ElementTree as et
from ekphrasis.classes.tokenizer import SocialTokenizer
from torch.utils.data import Dataset
from sklearn import preprocessing
from nlp import vectorize

import torch

torch.manual_seed(1)


class Sanitize():

    def __init__(self,text_path):
        self.text_path = text_path

    def sanitize(self, test = False):
        tree = et.parse(self.text_path)
        root = tree.getroot()
        critiques = []
        critique_to_opinion = []
        for review in root:
            for sentences in review:
                for sentence in sentences:
                    opinions = sentence.find("Opinions")
                    if test:
                        critiques.append(sentence.find("text").text)
                    if opinions:
                        if not test:
                            critiques.append(sentence.find("text").text)
                        critique_to_opinion.append([])
                        for opinion in opinions:
                            tag = opinion.attrib
                            critique_to_opinion[-1].append(tag)

        return critiques, critique_to_opinion

    def vectorsforcritiques(self, critiques, vocab, vec, max_words):
        vector_size = vec[-1].shape[0]
        zero_pad = torch.zeros((1, vector_size))
        unknown_vector = torch.randn((1, vector_size))*0.01
        unknown_vector = unknown_vector
        crit_vecs = []
        max = max_words
        for critique in critiques:
            crit_vecs.append([])
            for id, token in enumerate([y.lower() for y in SocialTokenizer(lowercase=True).tokenize(critique)], 1):
                try:
                    temp = vocab[token]
                    crit_vecs[-1].append(vec[temp].view(1, -1))
                except KeyError:
                    crit_vecs[-1].append(unknown_vector)
        for opinion_id, opinion in enumerate(crit_vecs):
            for iterator in range(len(opinion),max):
                crit_vecs[opinion_id].append(zero_pad)
        all = [token for vector in crit_vecs for token in vector]
        embeddings = torch.cat((all), 0)
        print("all of the sentences are shaped.")
        print(embeddings[80,:])
        print(embeddings.shape)
        return crit_vecs, embeddings

    def extract_from_text(self, opinions_sentences):
        all_categories, emotion_for_sentence = [] , []
        for i in opinions_sentences:
            all_categories.append([])
            emotion_for_sentence.append([])
            for dictionary in i:
                all_categories[-1].append(dictionary["category"])
                emotion_for_sentence[-1].append(dictionary["polarity"])
        return all_categories, emotion_for_sentence

    def get_max(self, all_sentences):
        len_sentence = []
        for sentence in all_sentences:
            len_sentence.append(len([y.lower() for y in SocialTokenizer(lowercase=True).tokenize(sentence)]))
        max_words = max(len_sentence)
        return max_words


class AspectDataset(Dataset):
    def __init__(self, word2idx, data, labels, true_labels):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            file (str): path to the data file
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
        """
        all = ["OTHER"]
        for i in labels:
            for element in i:
                all.append(element)
        all = list(set(all))
        for id1, lista in enumerate(true_labels):
            for id2, element in enumerate(lista):
                if element not in all:
                    true_labels[id1][id2] = "OTHER"

        self.word2idx = word2idx
        self.data = [SocialTokenizer(lowercase=True).tokenize(x) for x in data]
        self.labels = true_labels
        self.label_encoder = preprocessing.LabelBinarizer()
        self.label_encoder = self.label_encoder.fit(all)
        self.max_length = max([len(sentence) for sentence in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the returned dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['super', 'eagles', 'coach', 'sunday', 'oliseh',
                                    'meets', 'with', 'chelsea', "'", 's', 'victor',
                                    'moses', 'in', 'london', '<url>']
                self.target[index] = "neutral"

            the function will return:
            ::
                example = [  533  3908  1387   649 38127  4118    40  1876    63   106  7959 11520
                            22   888     7     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0]
                label = 1
        """
        sample, label = self.data[index], self.labels[index]
        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx, self.max_length)
        label = self.label_encoder.transform(label)

        from numpy import sum
        label = sum(label, 0)
        label[label > 1] = 1
        return sample, label, len(self.data[index]), index


class SentimentDataset(Dataset):
    def __init__(self, word2idx, data, true_labels, aspects_for_sentence):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            file (str): path to the data file
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
        """
        sentiment = ['positive', 'negative', 'neutral']
        self.word2idx = word2idx
        self.data = [SocialTokenizer(lowercase=True).tokenize(x) for x in data]
        self.max_length = max([len(sentence) for sentence in self.data])
        self.labels = true_labels
        self.label_encoder = preprocessing.LabelBinarizer()
        self.label_encoder = self.label_encoder.fit(sentiment)
        self.aspects_for_sentence = aspects_for_sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the returned dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['super', 'eagles', 'coach', 'sunday', 'oliseh',
                                    'meets', 'with', 'chelsea', "'", 's', 'victor',
                                    'moses', 'in', 'london', '<url>']
                self.target[index] = "neutral"

            the function will return:
            ::
                example = [  533  3908  1387   649 38127  4118    40  1876    63   106  7959 11520
                            22   888     7     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0]
                label = 1
        """
        sample, label = self.data[index], self.labels[index]
        getaspect = self.aspects_for_sentence[index]
        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx, self.max_length)
        label = self.label_encoder.transform([label])

        return sample, label, len(self.data[index]), index, getaspect


class MySentences(object):

    def __init__(self, dirname, quota, max_sentences):
        from math import ceil
        self.dirname = dirname
        self.positives = ceil(max_sentences*quota[2])
        self.negatives = ceil(max_sentences*quota[0])
        self.neutrals = max_sentences - self.positives - self.negatives
        print(self.positives, self.neutrals, self.negatives)
        self.max_sentences = max_sentences

    def get_sentiment(self):
        import gzip
        g = gzip.open(self.dirname, "r")
        sent_return, rating_ret, asp_return = [], [], []
        flag = False
        for l in g:
            sentences = [eval(l)["reviewText"]]
            ratings = ""
            sentiment = eval(l)["overall"]
            if sentiment < 3:
                ratings = ["negative"]
            elif sentiment > 3:
                ratings = ["positive"]
            else:
                ratings = ["neutral"]

            for sentence, rating in zip(sentences, ratings):
                    if rating == "positive" and self.positives != 0:
                        sent_return.append(sentence)
                        rating_ret.append(rating)
                        asp_return.append("OTHER")
                        self.positives -= 1
                    elif rating == "negative" and self.negatives != 0:
                        sent_return.append(sentence)
                        rating_ret.append(rating)
                        asp_return.append("OTHER")
                        self.negatives -= 1
                    elif rating == "neutral" and self.neutrals != 0:
                        sent_return.append(sentence)
                        rating_ret.append(rating)
                        asp_return.append("OTHER")
                        self.neutrals -= 1
                    print(self.positives, "/", self.negatives, "/", self.neutrals)
                    if self.neutrals + self.positives + self.negatives == 0:
                        flag = True
                        break

            if flag:
                break
        return sent_return, rating_ret, asp_return
