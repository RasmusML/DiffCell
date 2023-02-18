"""
Train a diffusion model and write sample images to jpg and npy files.
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

    #train_metadata = metadata[:3000]
    #train_metadata = stratify_metadata(metadata, 50)
    blacklist = [("Eg5 inhibitors", 0.1), ("Microtubule destabilizers", 0.3), ("Cholesterol-lowering", 6.0)]
    train_metadata = stratify_metadata(metadata, 60, blacklist=blacklist)

    logging.info("loading images")
    images = load_images_from_metadata(train_metadata, is_local)

    images = normalize_image_channel_wise(images)
    images = normalized_to_pseudo_zscore(images)

    logging.info("training")
    cropped_images = crop_images(images)

    batch_size = 32
    epochs = 600
    epoch_sample_times = 15

    if args.unconditional:
        train_diffusion_model(train_metadata, cropped_images, epochs = epochs, batch_size = batch_size, epoch_sample_times = epoch_sample_times)
    else:
        train_conditional_diffusion_model(train_metadata, cropped_images, epochs = epochs, batch_size = batch_size, epoch_sample_times = epoch_sample_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=False, action="store_true")
    parser.add_argument("--unconditional", default=False, action="store_true")

    args = parser.parse_args()

    main(args)