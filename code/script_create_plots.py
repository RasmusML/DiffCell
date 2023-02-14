"""
Generate a single plot of the images sampled at epoch 
intervals during the training of a diffusion model.
"""

import argparse
import logging

from plots import *
from dataset import *


def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    epoch_images, epochs = load_epoch_images(args.dir)
    plot_epoch_sample_series(epoch_images, epochs, path=join(args.dir, "epoch_samples.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", type=str, help="directory with epoch samples")

    args = parser.parse_args()

    main(args)

