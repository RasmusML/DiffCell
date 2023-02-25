"""
Sample images from a conditional diffusion model.
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

    whitelist = get_treatment_whitelist()

    compound_types = extract_compound_types(whitelist)
    concentration_types = extract_concentration_types(whitelist)

    compound_to_id, _ = get_label_mappings(compound_types)
    concentration_to_id, _ = get_label_mappings(concentration_types)

    unet = UNet_conditional(len(compound_types), len(concentration_types)).to(device)
    unet.load_state_dict(torch.load(args.pretrained, map_location=device))

    image_size = 64
    diffusion = Diffusion_conditional(img_size=image_size, noise_steps=1000, device=device)

    N_samples_per_treatment = args.N_treatment_samples
    treatments_to_sample = whitelist * N_samples_per_treatment

    compounds = [treatment[0] for treatment in treatments_to_sample]
    compounds = torch.from_numpy(np.array([compound_to_id[c] for c in compounds]))
    compounds = compounds.to(device)

    concentrations = [treatment[1] for treatment in treatments_to_sample]
    concentrations = torch.from_numpy(np.array([concentration_to_id[c] for c in concentrations]))
    concentrations = concentrations.to(device)

    batch_size = args.batch_size

    steps = np.arange(0, len(treatments_to_sample), batch_size)

    result_treatments = {}
    result_treatments["treatments"] = treatments_to_sample
    result_images = np.empty((len(treatments_to_sample), 3, image_size, image_size))

    for at in steps:
        logging.info(f"step {at} / {steps[-1]}")
        end = min(at + batch_size, len(treatments_to_sample))
        compounds_to_sample = compounds[at:end]
        concentrations_to_sample = concentrations[at:end]
        sampled_images = diffusion.sample(unet, N_images=len(compounds_to_sample), y_compounds=compounds_to_sample, y_concentrations=concentrations_to_sample)
        result_images[at:end] = sampled_images.detach().cpu().numpy()

    logging.info("done sampling")

    run_name = "DDPM_Conditional"
    os.makedirs(os.path.join("results", run_name, "sampling"), exist_ok=True)

    result_dir = os.path.join("results", run_name, "sampling")
    np.save(os.path.join(result_dir, "images.npy"), result_images)
    save_dict(result_treatments, os.path.join(result_dir, "metadata.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--N_treatment_samples", default=64, type=int)
    parser.add_argument("--pretrained", default="./results/DDPM_Conditional/weights/ckpt.pt", type=str)

    args = parser.parse_args()

    main(args)

