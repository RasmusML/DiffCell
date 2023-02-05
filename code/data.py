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
        images[i] = np.load(path).astype(np.float32).reshape(dims)
    
    return torch.from_numpy(images)

def load_image(path: str) -> torch.Tensor:
    img = torch.from_numpy(np.load(path).astype(np.float32))
    return image_view(img)

def _image_path(row: pd.Series) -> str:
    """ concats the multi cell folder name and file name """
    return "{}/{}".format(row["Multi_Cell_Image_Name"], row["Single_Cell_Image_Name"])


# --- Image transforms ---
def image_view(image: torch.Tensor) -> torch.Tensor:
    return image.reshape((3, 68, 68))

def normalize(image: torch.Tensor) -> torch.Tensor:
    return image / image.max()


