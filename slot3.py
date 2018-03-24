from preprocess import MySentences
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from helpers import progress, sort_cnn_batch, cnn_eval_dataset
from preprocess import SentimentDataset
from tools import set_logger, train_validation_split
from load_embeddings import load_word_vectors
from my_neural import CNNClassifier
import matplotlib.pyplot as plt
import copy
torch.manual_seed(1)

BATCH_SIZE = 15
EPOCHS = 13
vec_size = 300
datasets = "laptop"
default_embed_path = "word_embeds/amazon%s.txt" % vec_size
default_train_path = "train_data/reviews_Electronics_5.json.gz"

word2idx, idx2word, embeddings = load_word_vectors(default_embed_path, vec_size)
logging = set_logger("slot3.csv")
logging.debug("Epoch,Train acuracy, train f1, train loss, test accuracy, test f1, test Loss")
###############################################################################################

sentences = MySentences(default_train_path)
train_sentences, emotion_for_sentence, all_categories_train = [], [], []
for index, sentence in enumerate(sentences, 1):
    print("{} sentences".format(index))
    ts, efs, act = sentence
    train_sentences.append(ts)
    emotion_for_sentence.append(efs)
    all_categories_train.append(act)
    if index == 100:
        break
print(1)

import pickle
with open('dict.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
a_e = unserialized_data
a_e["OTHER"] = sum(a_e.values())/len(a_e) # mean of all vectors
print(2)
t_return, v_return = train_validation_split(0.3, train_sentences, emotion_for_sentence,
                                            all_categories_train)
print("split done")
train_sentences, emotion_for_sentence, all_categories_train = t_return[0], t_return[1], t_return[2]
validation_sentences, validation_emotion_for_sentence, all_categories_validation = v_return[0], v_return[1], v_return[2]
print(3)
train_set = SentimentDataset(word2idx, train_sentences,  emotion_for_sentence, all_categories_train)
validation_set = SentimentDataset(word2idx, validation_sentences,  validation_emotion_for_sentence, all_categories_validation)
print(4)
loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
loader_validation = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print(5)
_hparams = {
    "kernel_dim": 30,
    "kernel_sizes": (3, 4, 5),
    "dropout": 0.5,
    "output_size": 3,
    "trainable_emb": False,
    "noise": 0.,
    "aspect_embeddings": a_e
}

model = CNNClassifier(embeddings, **_hparams)

if torch.cuda.is_available():
    # recursively go over all modules
    # and convert their parameters and buffers to CUDA tensors
    model.cuda()
else:
    model.cpu()

parameters = filter(lambda p: p.requires_grad, model.parameters())
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters)


def train_epoch(_epoch, dataloader, model, loss_function):
    # switch to train mode -> enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    for i_batch, sample_batched in enumerate(dataloader, 1):

        # get the inputs (batch)
        inputs, labels, lengths, indices, get_aspect = sample_batched
        # in this point i have to concatenate the get_aspect embedding to the input.
        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels), get_aspect = sort_cnn_batch(lengths, (inputs, labels), get_aspect)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs, get_aspect)
        # 3 - compute loss
        _, labels = labels.squeeze().max(dim=1)
        loss = loss_function(outputs, labels)
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
    y[-1] = y[-1].reshape(-1, 3)
    y_hat[-1] = y_hat[-1].reshape(-1, 3)
    y1 = concatenate(y, axis=0)
    y2 = concatenate(y_hat, axis=0)
    ac_av = accuracy_score(y1, y2)
    f1_av = f1_score(y1, y2, average='macro')
    return ac_av, f1_av


train_dict = {
    "f1_scores": [],
    "accuracies": [],
    "avg_train_loss": []
}
val_dict = {
    "f1_scores": [],
    "accuracies": [],
    "avg_train_loss": []
}

for epoch in range(1, EPOCHS + 1):
    best_model_wts = copy.deepcopy(model.state_dict())
    # train the model for one epoch
    general_loss = train_epoch(epoch, loader_train, model, criterion)
    # evaluate the performance of the model, on both data sets
    avg_train_loss, (y, y_pred) = cnn_eval_dataset(loader_train, model, criterion)
    acc, f1 = av_metrics(y, y_pred)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_train_loss, acc, f1))
    #################################
    train_dict["f1_scores"].append(f1)
    train_dict["accuracies"].append(acc)
    train_dict["avg_train_loss"].append(avg_train_loss)
    #################################
    avg_val_loss, (y, y_pred) = cnn_eval_dataset(loader_validation, model, criterion)
    acc2, f12 = av_metrics(y, y_pred)
    print("\tDev:  loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_val_loss, acc2, f12))
    #################################
    val_dict["f1_scores"].append(f12)
    val_dict["accuracies"].append(acc2)
    val_dict["avg_train_loss"].append(avg_val_loss)
    #################################

    #################################
    logging.debug("{0},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f}".format(epoch, acc, f1, avg_train_loss,
                                                                                       acc2, f12, avg_val_loss,
                                                                                     ))


torch.save(model.state_dict(),"here.pt")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
epochs = list(range(1, EPOCHS+1))
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
plt.plot(epochs, train_dict['f1_scores'])
plt.plot(epochs, train_dict['accuracies'])
plt.plot(epochs, val_dict['f1_scores'])
plt.plot(epochs, val_dict['accuracies'])
plt.legend(['train f1', 'train accuracy', 'val f1', 'val accuracy'], loc='upper left')
plt.savefig("slot3 f1_accuracy.png")
plt.close()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_dict['avg_train_loss'])
plt.legend(['avg_train_loss', 'avg_test_loss', "model training"], loc='upper left')
plt.savefig("slot3 loss.png", bbox_inches='tight')
plt.close()
