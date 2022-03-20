from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

from utils.utils import read_all_datasets
from utils.utils import calculate_metrics
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import plot_pairwise

from augment import augment_train_set

import numpy as np

def augment_function(augment_algorithm_name, x_train, y_train, classes, N, limit_N=True):
    if augment_algorithm_name == 'as_dtw_dba_augment':
        return augment_train_set(x_train, y_train, classes, N,limit_N = limit_N,
                                 weights_method_name='as', distance_algorithm='dtw'), 'dtw'

def dba(X,y, classes, nb_prototypes):
    x_train = X
    y_train = y
    if len(x_train.shape) == 2:  # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # maximum number of prototypes which is the minimum count of a class
    # MAX_PROTOTYPES_PER_CLASS = 5
    # classes2, classes_counts = np.unique(y_train, return_counts=True)
    # max_prototypes = min(classes_counts.max() + 1, MAX_PROTOTYPES_PER_CLASS + 1)

    syn_train_set, distance_algorithm = augment_function('as_dtw_dba_augment', x_train, y_train, np.unique(y_train), nb_prototypes,limit_N=False)
    # get the synthetic train and labels
    syn_x_train, syn_y_train = syn_train_set
    # concat the synthetic with the reduced random train and labels
    # aug_x_train[:,:,col] = syn_x_train.tolist()
    # aug_y_train = syn_y_train

    # print(np.unique(y_train,return_counts=True))
    # print(np.unique(aug_y_train,return_counts=True))

    return (syn_x_train,syn_y_train)