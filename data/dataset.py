import os
import random
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import Sampler

from preprocessing.preprocess import FingerprintPreprocessor


class SiameseFingerprintDataset(Dataset):
    """
    A PyTorch Dataset class that generates pairs of fingerprint images on-the-fly
    for training a Siamese network with contrastive loss.
    """
    def __init__(self, file_data, transform=None, preprocessor=None):
        """
        Args:
            file_data (dict): A dictionary mapping (person_id, finger_name) to a list of file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            preprocessor (FingerprintPreprocessor, optional): The preprocessor for image normalization and resizing.
        """
        self.file_data = file_data
        self.preprocessor = preprocessor if preprocessor is not None else FingerprintPreprocessor(verbose=False)
        self.transform = transform if transform is not None else self.get_default_transform()

        # Pre-calculate lists for efficient sampling.
        self.finger_ids = list(self.file_data.keys())
        self.all_files = [item for sublist in self.file_data.values() for item in sublist]
        # Identify which fingers are eligible for creating positive pairs.
        self.eligible_positive_fids = [fid for fid in self.finger_ids if len(self.file_data[fid]) >= 2]

    @staticmethod
    def get_validation_transform():
        """Returns the standard torchvision transforms for validation and inference."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_training_transform():
        """Returns torchvision transforms with data augmentation for training."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """The length of the dataset is the total number of individual images."""
        return len(self.all_files)

    def _get_positive_pair(self):
        """Selects two different images from the same finger."""
        finger_id = random.choice(self.eligible_positive_fids)
        img_path1, img_path2 = random.sample(self.file_data[finger_id], 2)
        return img_path1, img_path2, 1.0

    def _get_negative_pair(self):
        """Selects one image from two different fingers."""
        finger_id1, finger_id2 = random.sample(self.finger_ids, 2)
        img_path1 = random.choice(self.file_data[finger_id1])
        img_path2 = random.choice(self.file_data[finger_id2])
        return img_path1, img_path2, 0.0

    def __getitem__(self, index):
        """
        Generates a single sample (a pair of images and a label).
        The pair can be positive (label=1.0, same finger) or negative (label=0.0, different fingers).
        """
        # Decide whether to generate a positive or negative pair with a 50/50 probability.
        # If no positive pairs can be created (e.g., in a sparse validation set),
        # this will always default to creating a negative pair.
        should_get_positive_pair = random.randint(0, 1) and self.eligible_positive_fids

        if should_get_positive_pair:
            img_path1, img_path2, label = self._get_positive_pair()
        else:
            img_path1, img_path2, label = self._get_negative_pair()

        # The preprocessor returns a numpy array (H, W, C), which ToTensor converts to (C, H, W).
        img1 = self.preprocessor.preprocess(img_path1)
        img2 = self.preprocessor.preprocess(img_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class TripletFingerprintDataset(Dataset):
    """
    A PyTorch Dataset class that generates triplets of fingerprint images on-the-fly
    (anchor, positive, negative) for training with TripletLoss.
    """
    def __init__(self, file_data, transform=None, preprocessor=None):
        """
        Args:
            file_data (dict): A dictionary mapping (person_id, finger_name) to a list of file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            preprocessor (FingerprintPreprocessor, optional): The preprocessor for image normalization and resizing.
        """
        self.file_data = file_data
        self.preprocessor = preprocessor if preprocessor is not None else FingerprintPreprocessor(verbose=False)
        self.transform = transform if transform is not None else self.get_training_transform()

        # We need fingers with at least two images to form a positive pair with an anchor.
        self.eligible_fids = [fid for fid, files in self.file_data.items() if len(files) >= 2]

        # Create a flat list of all possible anchor images from eligible fingers.
        self.anchor_files = [file for fid in self.eligible_fids for file in self.file_data[fid]]

        # Create a reverse map from file path to finger_id for efficient lookup.
        self.file_to_fid = {file: fid for fid, files in self.file_data.items() for file in files}

        self.finger_ids = list(self.file_data.keys())

    def __len__(self):
        """The length of the dataset is the number of possible anchor images."""
        return len(self.anchor_files)

    def __getitem__(self, index):
        """
        Generates a single triplet sample (anchor, positive, negative).
        """
        anchor_path = self.anchor_files[index]
        anchor_fid = self.file_to_fid[anchor_path]

        # --- Select the Positive ---
        # Get all images for the anchor's finger, excluding the anchor itself.
        positive_options = [p for p in self.file_data[anchor_fid] if p != anchor_path]
        positive_path = random.choice(positive_options)

        # --- Select the Negative ---
        # Choose a finger ID that is different from the anchor's finger ID.
        negative_fid = anchor_fid
        while negative_fid == anchor_fid:
            negative_fid = random.choice(self.finger_ids)

        negative_path = random.choice(self.file_data[negative_fid])

        # Preprocess and transform all three images
        anchor_img = self.transform(self.preprocessor.preprocess(anchor_path))
        positive_img = self.transform(self.preprocessor.preprocess(positive_path))
        negative_img = self.transform(self.preprocessor.preprocess(negative_path))

        return anchor_img, positive_img, negative_img


class FingerprintImageDataset(Dataset):
    """
    A PyTorch Dataset class that returns individual fingerprint images and their
    corresponding finger ID (as an integer label). This is used for online
    triplet mining where batches are constructed by a special sampler.
    """
    def __init__(self, file_data, transform=None, preprocessor=None):
        self.preprocessor = preprocessor or FingerprintPreprocessor(verbose=False)
        self.transform = transform or self.get_transform()

        # Create a flat list of (image_path, finger_id_index) and map fids to integers
        self.all_files = []
        self.finger_id_map = {fid: i for i, fid in enumerate(file_data.keys())}
        for fid, files in file_data.items():
            fid_index = self.finger_id_map[fid]
            for file_path in files:
                self.all_files.append((file_path, fid_index))

    @staticmethod
    def get_transform():
        """Returns the standard training transforms."""
        return SiameseFingerprintDataset.get_training_transform()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        img_path, label = self.all_files[index]
        img = self.preprocessor.preprocess(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class BalancedBatchSampler(Sampler):
    """
    A custom sampler that creates batches containing P classes (fingers)
    and K samples from each class. This is essential for online triplet mining.
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = np.array([item[1] for item in dataset.all_files])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset_len = len(dataset)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= self.dataset_len:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                start_idx = self.used_label_indices_count[class_]
                # Reshuffle and reset if we've run out of samples for this class
                if start_idx + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    start_idx = 0
                self.used_label_indices_count[class_] = start_idx + self.n_samples
                indices.extend(self.label_to_indices[class_][start_idx:start_idx + self.n_samples])
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.dataset_len // self.batch_size


def main():
    """
    Runs a demonstration of the SiameseFingerprintDataset.
    This function is executed when the script is run directly.
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

        # Instantiate the dataset.
        dataset = SiameseFingerprintDataset(train_files, preprocessor=FingerprintPreprocessor(verbose=False))

        logging.info(f"Dataset size: {len(dataset)}")

        # Retrieve and display information about a few random samples.
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
