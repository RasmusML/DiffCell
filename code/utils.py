import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
from sklearn import metrics

def save_dict(dictionary, file):
    with open(file, "wb") as f:
        pickle.dump(dictionary, f)


def load_dict(file):
    with open(file, 'rb') as f: 
         return pickle.load(f)


def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_files_in_dir(path: str):
    return [f for f in listdir(path) if isfile(join(path, f))]


def filter_file_extension(files: list[str], extension: str):
    return list(filter(lambda path: path.endswith(extension), files))


def get_label_mappings(labels: list[str]):
    label_to_id = {}
    id_to_label = {}

    for i, label in enumerate(labels):
        label_to_id[label] = i
        id_to_label[i] = label

    return label_to_id, id_to_label


def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(), 
        pred.detach().cpu().numpy(),
        normalize=normalize)


def normalize(matrix, axis):
    axis = {'true': 1, 'pred': 0}[axis]
    return matrix / matrix.sum(axis=axis, keepdims=True)
