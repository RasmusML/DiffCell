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
    small_metadata = metadata[:3000]

    images = load_images_from_metadata(small_metadata, is_local)

    images = normalize_channel_wise(images)
    images = normalized_to_zscore(images)

    cropped_images = view_cropped_images(images)
    train(cropped_images, epochs=20, batch_size=2)

