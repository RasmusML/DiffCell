from dataset import *
from models import *


def main():
    metadata = load_metadata()

    blacklist = []
    validation_metadata = stratify_metadata(metadata, 20, blacklist=blacklist)
    train_metadata = stratify_metadata(metadata, 420, blacklist=blacklist)

    _train_images = load_images_from_metadata(train_metadata)
    _train_images = normalize_image_channel_wise(_train_images)
    _train_images = normalized_to_pseudo_zscore(_train_images)
    train_images = crop_images(_train_images)

    _validation_images = load_images_from_metadata(validation_metadata)
    _validation_images = normalize_image_channel_wise(_validation_images)
    _validation_images = normalized_to_pseudo_zscore(_validation_images)
    validation_images = crop_images(_validation_images)

    train_VAE(train_images, validation_images)


if __name__ == "__main__":
    main()

