import numpy as np
import torch
import pandas as pd

# --- Metadata loading ---

# @Note: Relative to the root
LOCAL_DATASET_PATH = "./data/singlecell/"
SERVER_HOSTED_DATASET_PATH = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

def get_dataset_path(local_path: bool = True) -> str:
    return LOCAL_DATASET_PATH if local_path else SERVER_HOSTED_DATASET_PATH

def load_metadata(local_path: bool = True) -> pd.DataFrame:
    path = get_dataset_path(local_path)
    return pd.read_csv(path + "metadata.csv")

# --- Image loading ---

def load_images_from_metadata(metadata: pd.DataFrame, local_path: bool = True) -> torch.Tensor:
    path = get_dataset_path(local_path)
    paths = metadata.apply(lambda r: "{}/{}".format(path, _get_relative_image_path(r)), axis=1).tolist()
    return load_images_from_paths(paths)


def load_images_from_paths(paths: list[str]) -> torch.Tensor:
    dims = (3, 68, 68)
    images = np.zeros((len(paths), *dims), dtype=np.float32)
    
    for i, path in enumerate(paths):
        images[i] = np.load(path).astype(np.float32).transpose(2, 0, 1)
    
    return torch.from_numpy(images)


def load_image(path: str) -> torch.Tensor:
    np_image = np.load(path).astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(np_image)


def _get_relative_image_path(row: pd.Series) -> str:
    """ concats the multi cell folder name and file name """
    return "singh_cp_pipeline_singlecell_images/{}/{}".format(row["Multi_Cell_Image_Name"], row["Single_Cell_Image_Name"])


# --- Image normalization ---
def normalize(images: torch.Tensor) -> torch.Tensor:
    max_value_per_image, _ = images.flatten(start_dim=1).max(dim=1)
    img_tmp = images.flatten(start_dim=1) / max_value_per_image[:,None].expand(-1, 3*68*68)
    return img_tmp.reshape(images.shape)


def normalize_channel_wise(images: torch.Tensor) -> torch.Tensor:
    max_values, _ = images.flatten(start_dim=2).max(dim=2)
    img_tmp = images.flatten(start_dim=2) / max_values[:,:,None].expand(-1, 3, 68*68)
    return img_tmp.reshape(images.shape)


def normalize_constant(images: torch.Tensor) -> torch.Tensor:
    return images / 40_000

def normalized_to_zscore(images: torch.Tensor) -> torch.Tensor:
    """ [0,1] -> [-1,1] """
    return 2 * images.clamp(0, 1) - 1

def zscore_to_normalized(images: torch.Tensor) -> torch.Tensor:
    """ [-1,1] -> [0,1]"""
    return (images.clamp(-1,1) + 1) / 2

def view_cropped_images(images: torch.Tensor) -> torch.Tensor:
    return images[:,:,2:-2,2:-2]

# --- Metadata types ---
def get_all_MOA_types() -> np.ndarray:
    return np.array([
           'Actin disruptors', 'Aurora kinase inhibitors',
           'Cholesterol-lowering', 'DMSO', 'DNA damage', 'DNA replication',
           'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
           'Microtubule destabilizers', 'Microtubule stabilizers',
           'Protein degradation', 'Protein synthesis'])

def get_all_concentration_types():
    return np.array([0.0e+00, 1.0e-03, 3.0e-03, 1.0e-02, 3.0e-02, 1.0e-01, 3.0e-01,
       1.0e+00, 1.5e+00, 2.0e+00, 3.0e+00, 5.0e+00, 6.0e+00, 1.0e+01,
       1.5e+01, 2.0e+01, 3.0e+01, 5.0e+01, 1.0e+02])
