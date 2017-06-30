import numpy as np
from math import floor
from scipy import ndimage as nd


def color_codes():
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
    }
    return codes


def train_test_split(data, labels, test_size=0.1, random_state=42):
    # Init (Set the random seed and determine the number of cases for test)
    n_test = int(floor(data.shape[0]*test_size))

    # We create a random permutation of the data
    # First we permute the data indices, then we shuffle the data and labels
    np.random.seed(random_state)
    indices = np.random.permutation(range(0, data.shape[0])).tolist()
    np.random.seed(random_state)
    shuffled_data = np.random.permutation(data)
    np.random.seed(random_state)
    shuffled_labels = np.random.permutation(labels)

    x_train = shuffled_data[:-n_test]
    x_test = shuffled_data[-n_test:]
    y_train = shuffled_labels[:-n_test]
    y_test = shuffled_data[-n_test:]
    idx_train = indices[:-n_test]
    idx_test = indices[-n_test:]

    return x_train, x_test, y_train, y_test, idx_train, idx_test


def leave_one_out(data_list, labels_list):
    for i in range(0, len(data_list)):
        yield data_list[:i] + data_list[i+1:], labels_list[:i] + labels_list[i+1:], i


def nfold_cross_validation(data_list, labels_list, n=5, random_state=42, val_data=0.0):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(xrange(len(data_list)))

    for i in xrange(n):
        indices = shuffled_indices[i::n]
        tst_data = data_list[indices]
        tst_labels = labels_list[indices]
        tr_labels = labels_list[[idx for idx in shuffled_indices if idx not in indices]]
        tr_data = data_list[[idx for idx in shuffled_indices if idx not in indices]]
        val_len = int(len(tr_data) * val_data)
        yield tr_data[val_len:], tr_labels[val_len:], tr_data[:val_len], tr_labels[:val_len], tst_data, tst_labels


def get_biggest_region(labels):
    nu_labels = np.copy(labels)
    blobs, _ = nd.measurements.label(labels.astype(dtype=np.bool), nd.morphology.generate_binary_structure(3, 3))
    big_region = np.argmax(np.bincount(blobs.ravel())[1:])
    nu_labels[blobs != big_region + 1] = 0
    return nu_labels
