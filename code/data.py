import numpy as np
import torch
import pandas as pd

# --- Metadata loading ---
METADATA_PATH = "./data/singlecell/metadata.csv"

def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_PATH)


# --- Image loading ---
# @Note: relative to the root directory
IMAGES_DIR = "./data/singlecell/singh_cp_pipeline_singlecell_images"

def load_images_from_metadata(metadata: pd.DataFrame) -> torch.Tensor:
    paths = metadata.apply(lambda r: "{}/{}".format(IMAGES_DIR, _image_path(r)), axis=1).tolist()
    return load_images_from_paths(paths)

def load_images_from_paths(paths: list[str]) -> torch.Tensor:
    dims = (3, 68, 68)
    images = np.zeros((len(paths), *dims))
    
    for i, path in enumerate(paths):
        images[i] = np.load(path).astype(np.float32).transpose(2, 0, 1)
    
    return torch.from_numpy(images)

def load_image(path: str) -> torch.Tensor:
    img = torch.from_numpy(np.load(path).astype(np.float32))
    return image_view(img)

def _image_path(row: pd.Series) -> str:
    """ concats the multi cell folder name and file name """
    return "{}/{}".format(row["Multi_Cell_Image_Name"], row["Single_Cell_Image_Name"])


# --- Image transforms ---
def image_view(image: torch.Tensor) -> torch.Tensor:
    """
        68x68x3 -> 3x68x68
    """
    return image.permute(2, 0, 1)


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
