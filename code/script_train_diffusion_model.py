from dataset import *
from models import *
from plots import *

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            prog = 'diffusion model train script',
            description = 'Script to train a diffusion model')

    parser.add_argument("--server", action="store_true")
    parser.set_defaults(server=False)

    args = parser.parse_args()
    is_local = not args.server

    metadata = load_metadata(is_local)

    #train_metadata = metadata[:3000]
    train_metadata = stratify_metadata(metadata, 50)

    images = load_images_from_metadata(train_metadata, is_local)

    images = normalize_channel_wise(images)
    images = normalized_to_zscore(images)

    cropped_images = view_cropped_images(images)
    train(cropped_images, epochs=20, batch_size=2, epoch_sample_times=10)

