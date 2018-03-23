from preprocess import Sanitize
from my_neural import AttentiveRNN
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from helpers import progress, sort_batch, eval_dataset
from preprocess import AspectDataset
from tools import set_logger, train_validation_split
from load_embeddings import load_word_vectors

import matplotlib.pyplot as plt

BATCH_SIZE = 10
EPOCHS = 15

_hparams = {
    "rnn_size": 150,
    "rnn_layers": 1,
    "noise": 0.5,
    "dropout_words": 0.5,
    "bidirectional": True,
    "dropout_rnn": 0.5,
}

##############################################################################################
vec_size = 300

datasets = "laptop"
default_embed_path = "word_embeds/amazon%s.txt"%vec_size

if datasets == "rest":
    default_train_path = "train_data/ABSA16_Restaurants_Train_SB1_v2.xml"
    default_test_path = "test_data/EN_REST_SB1_TEST.xml.B"
else:
    default_train_path = "train_data/train_english_german1.xml"
    default_test_path = "test_data/EN_LAPT_SB1_TEST_.xml.B"

# get training data and transform them to word vectors
cleared = Sanitize(default_train_path)
word2idx, idx2word, embeddings = load_word_vectors(default_embed_path, vec_size)
logging = set_logger("%s %s embeddings.csv" % (datasets, vec_size))
logging.debug("{} , {}".format(_hparams, BATCH_SIZE))
logging.debug("Epoch,Train acuracy, train f1, train loss, test accuracy, test f1, test Loss, general loss")
###############################################################################################

train_sentences, train_opinions_sentences = cleared.sanitize()  # critique and opinion to said critique.
all_categories_train, emotion_for_sentence = cleared.extract_from_text(train_opinions_sentences)
test_cleared = Sanitize(default_test_path)
test_sentences, test_opinions_sentences = test_cleared.sanitize()
max_words = cleared.get_max(train_sentences + test_sentences)
# TEST DATA

# from collections import Counter
# occurences = Counter([x for element in all_categories_train for x in element])
# all_categories_train = size_reduction(all_categories_train, occurences, 0)

all_categories_test, _ = cleared.extract_from_text(test_opinions_sentences)
pure_all = all_categories_train.copy()
train_return, validation_return = train_validation_split(0.3, train_sentences, all_categories_train)
train_sentences, all_categories_train = train_return[0], train_return[1]
val_sentences, all_categories_val = validation_return[0], validation_return[1]

train_set = AspectDataset(word2idx, train_sentences, pure_all, all_categories_train)
validation_set = AspectDataset(word2idx, val_sentences, pure_all, all_categories_val)
test_set = AspectDataset(word2idx, test_sentences, pure_all, all_categories_test)

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_validation = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

classes = len(train_set.label_encoder.classes_)

model = AttentiveRNN(embeddings, nclasses=classes, **_hparams)

if torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()

parameters = filter(lambda p: p.requires_grad, model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(parameters)


def train_epoch(_epoch, dataloader, model, loss_function):
    # switch to train mode -> enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    for i_batch, sample_batched in enumerate(dataloader, 1):

        # get the inputs (batch)
        inputs, labels, lengths, indices = sample_batched

        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels) = sort_batch(lengths, (inputs, labels))

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            lengths = Variable(lengths.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            lengths = Variable(lengths)
        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs, lengths)
        # 3 - compute loss
        loss = loss_function(outputs, labels.float())
        loss.backward()

        # 5 - update weights
        optimizer.step()

        running_loss += loss.data[0]

        # print statistics
        progress(loss=loss.data[0],
                 epoch=_epoch,
                 batch=i_batch,
                 batch_size=BATCH_SIZE,
                 dataset_size=len(train_set))
    return loss.data[0]


#############################################################
# Train
#############################################################


def av_metrics(y, y_hat):
    from numpy import concatenate
    from sklearn.metrics import accuracy_score, f1_score
    y1 = concatenate(y, axis=0)
    y2 = concatenate(y_hat, axis=0)
    ac_av = accuracy_score(y1, y2)
    f1_av = f1_score(y1, y2, average='micro')
    return ac_av, f1_av


train_dict = {
    "f1_scores": [],
    "accuracies": [],
    "avg_train_loss": []
}
test_dict = {
    "f1_scores": [],
    "accuracies": [],
    "avg_train_loss": []
}
val_dict = {
    "f1_scores": [],
    "accuracies": [],
    "avg_train_loss": []
}

train_losses = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    general_loss = train_epoch(epoch, loader_train, model, criterion)
    train_losses.append(general_loss)
    # evaluate the performance of the model, on both data sets
    avg_train_loss, (y, y_pred) = eval_dataset(loader_train, model, criterion)
    acc, f1 = av_metrics(y, y_pred)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_train_loss,acc,f1))
    #################################
    train_dict["f1_scores"].append(f1)
    train_dict["accuracies"].append(acc)
    train_dict["avg_train_loss"].append(avg_train_loss)
    #################################
    avg_val_loss, (y, y_pred) = eval_dataset(loader_validation, model, criterion)
    acc2, f12 = av_metrics(y, y_pred)
    print("\tDev:  loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_val_loss, acc2, f12))
    #################################
    val_dict["f1_scores"].append(f12)
    val_dict["accuracies"].append(acc2)
    val_dict["avg_train_loss"].append(avg_val_loss)
    #################################
    avg_test_loss, (ty, ty_pred) = eval_dataset(loader_test, model, criterion)
    tacc2, tf12 = av_metrics(ty, ty_pred)
    print("\tTest:  loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_test_loss, tacc2, tf12))
    #################################
    test_dict["f1_scores"].append(tf12)
    test_dict["accuracies"].append(tacc2)
    test_dict["avg_train_loss"].append(avg_test_loss)

    logging.debug("{0},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f}".format(epoch, acc, f1, avg_train_loss,
                                                                                       acc2, f12, avg_val_loss,
                                                                                       general_loss))

#  plotting!!
plt.xlabel("Epochs")
plt.ylabel("Metrics")
epochs = list(range(1, EPOCHS+1))
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
plt.plot(epochs, train_dict['f1_scores'])
plt.plot(epochs, train_dict['accuracies'])
plt.plot(epochs, val_dict['f1_scores'])
plt.plot(epochs, val_dict['accuracies'])
plt.legend(['train f1', 'train accuracy', 'val f1', 'val accuracy'], loc='upper left')
plt.savefig("f1_accuracy.png")
plt.close()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_dict['avg_train_loss'])
plt.plot(epochs, val_dict['avg_train_loss'])
plt.plot(epochs, train_losses)
plt.legend(['avg_train_loss', 'avg_val_loss', "model training"], loc='upper left')
plt.savefig("loss.png", bbox_inches='tight')
plt.close()