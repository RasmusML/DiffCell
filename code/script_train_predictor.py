"""
Train a classifier model
"""

from dataset import *
from plots import *
from utils import *
from models import *

import argparse
import logging

def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()

    logging.info("loading data")
    is_local = not args.server
    metadata = load_metadata(is_local)

    validation_metadata = stratify_metadata(metadata, 20)
    train_metadata = stratify_metadata(metadata, 3000).drop(validation_metadata.index, errors="ignore")

    _train_images = load_images_from_metadata(train_metadata, is_local)
    _train_images = normalize_image_channel_wise(_train_images)
    _train_images = normalized_to_pseudo_zscore(_train_images)
    train_images = crop_images(_train_images)

    _validation_images = load_images_from_metadata(validation_metadata, is_local)
    _validation_images = normalize_image_channel_wise(_validation_images)
    _validation_images = normalized_to_pseudo_zscore(_validation_images)
    validation_images = crop_images(_validation_images)

    #train_MOA_classifier(train_metadata, train_images, validation_metadata, validation_images)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument("model", choices=["compound", "concentration"])
    parser.add_argument("--server", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
