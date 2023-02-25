"""
Sample images  + latent representation from a VAE.
"""

from dataset import *
from plots import *
from utils import *
from models import *

import os
import argparse
import logging


def main(args):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    fix_seed()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    logging.info("loading data")
    is_local = not args.server
    metadata = load_metadata(is_local)

    whitelist = get_treatment_whitelist()
    blacklist = get_treatment_blacklist()
    selected = [treatment for treatment in whitelist if treatment not in blacklist]
    train_metadata = stratify_metadata(metadata, args.N_treatment_samples, whitelist=selected)

    logging.info("loading images")
    images = load_images_from_metadata(train_metadata, is_local)

    images = normalize_image_channel_wise(images)
    images = normalized_to_pseudo_zscore(images)

    latent_features = 256
    image_size = 64
    vae = CytoVariationalAutoencoder(np.array([3, image_size, image_size]), latent_features)
    vae.load_state_dict(torch.load("./results/VAE_predictor/weights/ckpt.pt", map_location=torch.device('cpu')))
    vae.eval()

    batch_size = args.batch_size

    steps = np.arange(0, len(images), batch_size)

    result_treatments = {}
    result_treatments["treatments"] = train_metadata
    result_images = np.empty((len(images), latent_features))

    for at in steps:
        logging.info(f"step {at} / {steps[-1]}")
        end = min(at + batch_size, len(images))
        vae_parse = vae(images[at:end])
        latent_representations = vae_parse["z"]
        z = latent_representations.detach().cpu().numpy()
        result_images[at:end] = z

    logging.info("done sampling")

    run_name = "VAE_predictor"
    os.makedirs(os.path.join("results", run_name, "sampling"), exist_ok=True)

    result_dir = os.path.join("results", run_name, "sampling")
    np.save(os.path.join(result_dir, "images.npy"), result_images)
    save_dict(result_treatments, os.path.join(result_dir, "metadata.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--N_treatment_samples", default=400, type=int)
    parser.add_argument("--pretrained", default="./results/VAE_predictor/weights/ckpt600.pt", type=str)
    parser.add_argument("--server", default=False, action="store_true")

    args = parser.parse_args()

    main(args)

