import sys

import math
import torch
from torch.autograd import Variable


def progress(loss, epoch, batch, batch_size, dataset_size):
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def eval_dataset(dataloader, model, loss_function):
    # switch to eval mode -> disable regularization layers, such as Dropout
    model.eval()

    y_pred = []
    y = []

    total_loss = 0
    for i_batch, sample_batched in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths, indices = sample_batched
        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels) = sort_batch(lengths, (inputs, labels))

        # convert to CUDA Variables
        # if torch.cuda.is_available():
        #     inputs = Variable(inputs.cuda())
        #     labels = Variable(labels.cuda())
        #     lengths = Variable(lengths.cuda())
        # else:
        #     inputs = Variable(inputs)
        #     labels = Variable(labels)
        #     lengths = Variable(lengths)

        outputs = model(inputs, lengths)
        loss = loss_function(outputs, labels.float())
        total_loss += loss.data[0]
        #_, predicted = torch.max(outputs.data, 1)
        predicted = (torch.sigmoid(outputs)>=0.5).long()
        y.append(labels.data.cpu().numpy())
        y_pred.append(predicted.data.cpu().numpy())

    avg_loss = total_loss / i_batch

    return avg_loss, (y, y_pred)


def sort_batch(lengths, others):
    """
    Sort batch data and labels by length
    Args:
        lengths (nn.Tensor): tensor containing the lengths for the data

    Returns:

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
    sorted_lengths = sorted_lengths[reverse_idx]

    return sorted_lengths, (lst[sorted_idx][reverse_idx] for lst in others)


def cnn_eval_dataset(dataloader, model, loss_function):
    # switch to eval mode -> disable regularization layers, such as Dropout
    model.eval()

    y_pred = []
    y = []

    total_loss = 0
    for i_batch, sample_batched in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths, indices, get_aspect = sample_batched
        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels), get_aspect = sort_cnn_batch(lengths, (inputs, labels), get_aspect)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        outputs = model(inputs, get_aspect)
        # outputs = torch.nn.Softmax()(outputs) #this is not needed at all , but for the sake of symmetry i put it.
        _, true_labels = labels.max(dim=2)
        loss = loss_function(outputs, true_labels.squeeze())
        total_loss += loss.data[0]
        view = outputs.view(-1, 3)
        pred = (view == view.max(dim=1, keepdim=True)[0]).view_as(outputs)
        predicted = pred.long()
        labels = labels.data.cpu().squeeze()
        y.append(labels.numpy())
        y_pred.append(predicted.data.cpu().numpy())

    avg_loss = total_loss / i_batch

    return avg_loss, (y, y_pred)



def sort_cnn_batch(lengths, others, aspect):
    """
    Sort batch data and labels by length
    Args:
        lengths (nn.Tensor): tensor containing the lengths for the data

    Returns:

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
    sorted_lengths = sorted_lengths[reverse_idx]
    aspect_return = [aspect[idx] for idx in reverse_idx]

    return sorted_lengths, (lst[sorted_idx][reverse_idx] for lst in others), aspect_return
