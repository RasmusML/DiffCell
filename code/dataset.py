import numpy as np
import torch
import pandas as pd
import itertools

from typing import List, Tuple

from utils import *

# --- Metadata loading ---

LOCAL_DATASET_PATH = "data/singlecell/"
SERVER_HOSTED_DATASET_PATH = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

def get_dataset_path(local_path: bool = True) -> str:
    return LOCAL_DATASET_PATH if local_path else SERVER_HOSTED_DATASET_PATH

def load_metadata(local_path: bool = True) -> pd.DataFrame:
    path = get_dataset_path(local_path)
    return pd.read_csv(path + "metadata.csv")

# --- Image loading ---
def load_images_from_metadata(metadata: pd.DataFrame, local_path: bool = True) -> torch.Tensor:
    """
        out: N x c x h x w
    """
    path = get_dataset_path(local_path)
    paths = metadata.apply(lambda r: "{}/{}".format(path, _get_relative_image_path(r)), axis=1).tolist()

    dim = (3, 68, 68)
    images = np.zeros((len(paths), *dim), dtype=np.float32)
    
    for i, path in enumerate(paths):
        images[i] = np.load(path).astype(np.float32).transpose(2, 0, 1)
    
    return torch.from_numpy(images)


def stratify_metadata(metadata: pd.DataFrame, images_per_moa=100,
                      whitelist: List[Tuple[str, int]] | None = None) -> pd.DataFrame:

    if whitelist:
        treatments = whitelist
    else:
        compounds = get_all_compound_types()
        concentration = get_all_concentration_types()
        treatments = list(itertools.product(compounds, concentration))

    groups = metadata.groupby(by=["Image_Metadata_Compound", "Image_Metadata_Concentration"])
    stratified = pd.DataFrame(columns=metadata.columns)

    for treatment in treatments:
        try:
            group = groups.get_group(treatment)

            if not isinstance(group, pd.DataFrame):
                raise Exception("Group is not a DataFrame")

            stratified = pd.concat([stratified, group[:images_per_moa]])

        except Exception:
            pass # treatment combination not in metadata

    return stratified


def _get_relative_image_path(row: pd.Series) -> str:
    """ concats the multi cell folder name and file name """
    return "singh_cp_pipeline_singlecell_images/{}/{}".format(row["Multi_Cell_Image_Name"], row["Single_Cell_Image_Name"])


# --- Image normalization ---
def normalize_image(images: torch.Tensor) -> torch.Tensor:
    max_value_per_image, _ = images.flatten(start_dim=1).max(dim=1)
    img_tmp = images.flatten(start_dim=1) / max_value_per_image[:,None].expand(-1, 3*68*68)
    return img_tmp.reshape(images.shape)


def normalize_image_channel_wise(images: torch.Tensor) -> torch.Tensor:
    max_values, _ = images.flatten(start_dim=2).max(dim=2)
    img_tmp = images.flatten(start_dim=2) / max_values[:,:,None].expand(-1, 3, 68*68)
    return img_tmp.reshape(images.shape)


def normalize_image_by_constant(images: torch.Tensor) -> torch.Tensor:
    return images / 40_000


def normalized_to_pseudo_zscore(images: torch.Tensor) -> torch.Tensor:
    """ [0,1] -> [-1,1] """
    return 2 * images.clamp(0, 1) - 1


def pseudo_zscore_to_normalized(images: torch.Tensor) -> torch.Tensor:
    """ [-1,1] -> [0,1]"""
    return (images.clamp(-1,1) + 1) / 2


def crop_images(images: torch.Tensor) -> torch.Tensor:
    return images[:,:,2:-2,2:-2]


# --- Metadata types ---
def get_all_MOA_types() -> np.ndarray:
    return np.array([
           'Actin disruptors', 'Aurora kinase inhibitors',
           'Cholesterol-lowering', 'DMSO', 'DNA damage', 'DNA replication',
           'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
           'Microtubule destabilizers', 'Microtubule stabilizers',
           'Protein degradation', 'Protein synthesis'])


def get_all_concentration_types() -> np.ndarray:
    return np.array([0.0e+00, 1.0e-03, 3.0e-03, 1.0e-02, 3.0e-02, 1.0e-01, 3.0e-01,
       1.0e+00, 1.5e+00, 2.0e+00, 3.0e+00, 5.0e+00, 6.0e+00, 1.0e+01,
       1.5e+01, 2.0e+01, 3.0e+01, 5.0e+01, 1.0e+02])


def get_all_compound_types() -> np.ndarray:
    return np.array(['DMSO', 'taxol', 'AZ138', 'AZ-U', 'cytochalasin B', 'nocodazole',
       'AZ-A', 'latrunculin B', 'epothilone B', 'colchicine',
       'cytochalasin D', 'ALLN', 'methotrexate', 'MG-132', 'vincristine',
       'AZ-C', 'AZ841', 'etoposide', 'demecolcine', 'emetine',
       'cisplatin', 'chlorambucil', 'anisomycin', 'cyclohexamide',
       'AZ258', 'mitomycin C', 'AZ-J', 'lactacystin', 'docetaxel',
       'proteasome inhibitor I', 'bryostatin', 'PD-169316',
       'alsterpaullone', 'camptothecin', 'PP-2', 'mevinolin/lovastatin',
       'floxuridine', 'simvastatin', 'mitoxantrone'], dtype=object)


def get_treatment_whitelist():
    return [
        ("AZ138", 0.1), ("AZ138", 0.3), ("AZ138", 1),
        ("AZ-A", 0.1), ("AZ-A", 0.3), ("AZ-A", 1),
        ("epothilone B", 0.1), ("epothilone B", 0.3), ("epothilone B", 1),
        ("vincristine", 0.1), ("vincristine", 0.3), ("vincristine", 1),
        ("AZ-C", 0.1), ("AZ-C", 0.3), ("AZ-C", 1),
        ("AZ841", 0.1), ("AZ841", 0.3), ("AZ841", 1),
        ("emetine", 0.1), ("emetine", 0.3), ("emetine", 1),
        ("AZ258", 0.1), ("AZ258", 0.3), ("AZ258", 1),
        ("mitomycin C", 0.1), ("mitomycin C", 0.3), ("mitomycin C", 1),
        ("DMSO", 0), # no treatment
    ]

def extract_compound_types(treatments):
    compounds = list(set([treatment[0] for treatment in treatments]))
    compounds.sort()
    return compounds


def extract_concetration_types(treatments):
    concentration = list(set([treatment[1] for treatment in treatments]))
    concentration.sort()
    return concentration


def get_treatment_blacklist():
    return [
        ("AZ138", 0.3),
        ("AZ-A", 0.3),
        ("epothilone B", 0.3),
        ("vincristine", 0.3),
    ]


# --- Loading result images ---
def load_epoch_images(epoch_image_dir: str):
    file_names = get_files_in_dir(epoch_image_dir)
    npy_file_names = filter_file_extension(file_names, ".npy")
    paths = [join(epoch_image_dir, image_name) for image_name in npy_file_names]
    
    epoch_arr = np.array([int(name.split(".")[0]) for name in npy_file_names], dtype=np.int32)
    epoch_ordering = np.argsort(epoch_arr)
    
    image_list = [np.load(path) for path in paths]
    images = np.array([image_list[i] for i in epoch_ordering])
    
    epoch_arr.sort()
    
    return torch.from_numpy(images), torch.from_numpy(epoch_arr)

