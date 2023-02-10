import torch
import numpy as np

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

from os import listdir
from os.path import isfile, join

def get_files_in_dir(path: str):
    return [f for f in listdir(path) if isfile(join(path, f))]

def filter_file_extension(files: list[str], extension: str):
    return list(filter(lambda path: path.endswith(extension), files))


