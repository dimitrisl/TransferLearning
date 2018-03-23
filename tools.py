import os


def set_logger(filename='notes.log'):
    """
    Function that instantiates
    :param filename:
    :return:
    """
    import logging
    log_formatter = logging.Formatter("%(message)s")
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(logging.DEBUG)
    if root_logger.handlers:
        root_logger.handlers = []
    # setting file handler for logger
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    return root_logger


def size_reduction(sentences, dictionary, threshold):
    pairs = zip(dictionary.values(),dictionary.keys())
    remove = []
    for occurences, word in pairs:
        if occurences < threshold:
            remove.append(word)
    for id1, aspects in enumerate(sentences):
        for id2, aspect in enumerate(aspects):
            if aspect in remove:
                sentences[id1][id2] = "OTHER"

    return sentences


def train_validation_split(percentage, *aligned_lists):
    from sklearn.model_selection import train_test_split
    indices = list(range(len(aligned_lists[0])))
    train_indices, validation_indices = train_test_split(indices, test_size=percentage, random_state=1)
    train_return, validation_return = [], []
    for index,category in enumerate(aligned_lists):
        print("%s / %s"%(index, len(aligned_lists)))
        train_return.append([])
        validation_return.append([])
        for index in indices:
            if index in train_indices:
                train_return[-1].append(category[index])
            else:
                validation_return[-1].append(category[index])
    return train_return, validation_return
