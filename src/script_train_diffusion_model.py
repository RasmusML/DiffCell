from models import *
from dataset import *
from plots import *
from utils import *

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            prog = 'diffusion model train script',
            description = 'Script to train a diffusion model')

    parser.add_argument("--server", action="store_true")
    parser.set_defaults(server=False)

    parser.add_argument("--unconditional", action="store_true")
    parser.set_defaults(unconditional=False)

    args = parser.parse_args()
    is_local = not args.server

    fix_seed()

    print("loading data")
    metadata = load_metadata(is_local)

    #train_metadata = metadata[:3000]
    #train_metadata = stratify_metadata(metadata, 50)
    blacklist = [("Eg5 inhibitors", 0.1), ("Microtubule destabilizers", 0.3), ("Cholesterol-lowering", 6.0)]
    train_metadata = stratify_metadata(metadata, 60, blacklist=blacklist)

    print("loading images")
    images = load_images_from_metadata(train_metadata, is_local)

    images = normalize_channel_wise(images)
    images = normalized_to_zscore(images)

    print("training")
    cropped_images = view_cropped_images(images)

    if args.unconditional:
        train_diffusion_model(train_metadata, cropped_images, epochs=600, batch_size=6, epoch_sample_times=15)
    else:
        train_conditional_diffusion_model(train_metadata, cropped_images, epochs=600, batch_size=6, epoch_sample_times=15)
