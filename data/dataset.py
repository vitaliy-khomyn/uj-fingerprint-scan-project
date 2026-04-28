import logging
import os
import random
import torch
from preprocessing.preprocess import FingerprintPreprocessor
from torch.utils.data import Dataset
from torchvision import transforms


class SiameseFingerprintDataset(Dataset):
    """
    A PyTorch Dataset class that generates pairs of fingerprint images on-the-fly
    for training a Siamese network with contrastive loss.
    """
    def __init__(self, file_data, transform=None, preprocessor=None, use_cache=False):
        """
        Initializes the dataset with file data, transforms, and options.

        Args:
            file_data (dict): A dictionary mapping (person_id, finger_name) to a list of file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            preprocessor (FingerprintPreprocessor, optional): The preprocessor for image normalization and resizing.
            use_cache (bool): If True, pre-loads all preprocessed images into RAM.

        Returns:
            None
        """
        self.file_data = file_data
        self.preprocessor = preprocessor if preprocessor is not None else FingerprintPreprocessor(verbose=False)
        self.transform = transform if transform is not None else self.get_default_transform()
        self.use_cache = use_cache

        # pre-calculate list for efficient sampling
        self.finger_ids = list(self.file_data.keys())
        self.all_files = [item for sublist in self.file_data.values() for item in sublist]
        # identify which fingers are eligible for creating positive pairs.
        self.eligible_positive_fids = [fid for fid in self.finger_ids if len(self.file_data[fid]) >= 2]

        self.cache = {}
        if self.use_cache:
            # cache preprocessed images in memory to eliminate disk I/O bottlenecks during training.
            logging.info(f"Caching {len(self.all_files)} images in memory...")
            for file_path in self.all_files:
                self.cache[file_path] = self.preprocessor.preprocess(file_path)
        else:
            logging.info("In-memory caching is disabled to save RAM. Images will be loaded on-the-fly.")

    @staticmethod
    def get_validation_transform():
        """Returns the standard torchvision transforms for validation and inference."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_training_transform():
        """
        Returns torchvision transforms with data augmentation for training.

        Returns:
            transforms.Compose: The training transformations.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """
        Calculates the total length of the dataset.

        Returns:
            int: The total number of individual images.
        """
        return len(self.all_files)

    def _get_positive_pair(self):
        """
        Selects two different images from the same finger.

        Returns:
            tuple: A tuple containing two image paths and a positive label (1.0).
        """
        finger_id = random.choice(self.eligible_positive_fids)
        img_path1, img_path2 = random.sample(self.file_data[finger_id], 2)
        return img_path1, img_path2, 1.0

    def _get_negative_pair(self):
        """
        Selects one image from two different fingers.

        Returns:
            tuple: A tuple containing two image paths and a negative label (0.0).
        """
        finger_id1, finger_id2 = random.sample(self.finger_ids, 2)
        img_path1 = random.choice(self.file_data[finger_id1])
        img_path2 = random.choice(self.file_data[finger_id2])
        return img_path1, img_path2, 0.0

    def __getitem__(self, index):
        """
        Generates a single sample (a pair of images and a label).
        The pair can be positive (label=1.0, same finger) or negative (label=0.0, different fingers).

        Args:
            index (int): The index of the item.

        Returns:
            tuple: A tuple containing two processed image tensors and a label tensor.
        """
        # Decide whether to generate a positive or negative pair with a 50/50 probability.
        # If no positive pairs can be created (e.g., in a sparse validation set),
        # this will always default to creating a negative pair.
        should_get_positive_pair = random.randint(0, 1) and self.eligible_positive_fids

        if should_get_positive_pair:
            img_path1, img_path2, label = self._get_positive_pair()
        else:
            img_path1, img_path2, label = self._get_negative_pair()

        # preprocessor returns a numpy array (H, W, C), which ToTensor converts to (C, H, W).
        if self.use_cache:
            img1 = self.cache[img_path1]
            img2 = self.cache[img_path2]
        else:
            img1 = self.preprocessor.preprocess(img_path1)
            img2 = self.preprocessor.preprocess(img_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def main():
    """
    Runs a demonstration of the SiameseFingerprintDataset.
    This function is executed when the script is run directly.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from data.data_loader import load_and_split_data

    try:
        DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
        if not DATASET_PATH:
            raise ValueError("SOCOFING_DATASET_PATH environment variable not set.")

        logging.info("Loading data to demonstrate dataset functionality...")
        train_files, _ = load_and_split_data(DATASET_PATH)

        if not train_files:
            logging.warning("No training files were loaded. Cannot demonstrate dataset.")
            return

        # instantiate the dataset.
        dataset = SiameseFingerprintDataset(train_files, preprocessor=FingerprintPreprocessor(verbose=False))

        logging.info(f"Dataset size: {len(dataset)}")

        # retrieve and display information about a few random samples.
        for i in range(3):
            img1, img2, label = dataset[random.randint(0, len(dataset) - 1)]
            label_str = 'Positive' if label.item() == 1.0 else 'Negative'

            logging.info(f"\n--- Sample {i+1} ---")
            logging.info(f"  Pair Type: {label_str} (Label: {label.item()})")
            logging.info(f"  Image 1 Tensor Shape: {img1.shape}")
            logging.info(f"  Image 2 Tensor Shape: {img2.shape}")

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"\nERROR: {e}")
        logging.error("Please ensure your .env file is set up correctly.")


if __name__ == '__main__':
    main()
