import logging
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from typing import Any, List, Optional

from data.data_loader import load_and_split_data
from data.dataset import SiameseFingerprintDataset
from models.model import EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor

# configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("results", "trained_models", "embedding_net.pth"))
ENROLLMENT_DB_PATH = os.getenv("ENROLLMENT_DB_PATH", os.path.join("results", "enrollments", "enrollment_db.pkl"))
ENROLL_USERS_START = 401
ENROLL_USERS_END = 500


def generate_user_template(file_paths: List[str], embedding_net: torch.nn.Module, preprocessor: FingerprintPreprocessor, transform: Any, device: torch.device, finger_id: tuple) -> Optional[np.ndarray]:
    """
    Processes files and generates a normalized mean embedding template sequence for a target user.

    Args:
        file_paths (List[str]): List of absolute system references dynamically bound.
        embedding_net (torch.nn.Module): Current generated network model state dict executed.
        preprocessor (FingerprintPreprocessor): Transformer mapping initialization rules.
        transform (Any): Target conversion metric transformations applied.
        device (torch.device): Device configuration target hardware reference.
        finger_id (tuple): Assigned fingerprint designation index key mapping array.

    Returns:
        Optional[np.ndarray]: Processed sequence target float dimensions explicitly aggregated globally.
    """
    embeddings = []
    for img_path in file_paths:
        try:
            # preprocess and transform image
            processed_img = preprocessor.preprocess(img_path)
            tensor_img = transform(processed_img).unsqueeze(0).to(device)

            embedding = embedding_net(tensor_img)
            embeddings.append(embedding.cpu().numpy())
        except Exception as e:
            logging.warning(f"Could not process file {img_path}. Error: {e}")

    if not embeddings:
        logging.warning(f"No valid embeddings generated for {finger_id}. Skipping.")
        return None

    # calculate mean embedding for finger
    mean_embedding = np.mean(embeddings, axis=0)
    # l2-normalize final template vector
    norm = np.linalg.norm(mean_embedding)
    return mean_embedding / norm if norm != 0 else mean_embedding


def main() -> None:
    """
    Executes enrollment execution logically registering genuine elements iteratively.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # check global environment handling exception cases
    if not DATASET_PATH:
        raise ValueError("SOCOFING_DATASET_PATH environment variable not set. Please create a .env file.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # load model
    logging.info("Loading trained embedding model...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    embedding_net.load_state_dict(checkpoint['model_state_dict'])
    embedding_net.eval()
    logging.info("Model loaded successfully.")

    # load data
    logging.info("Loading data for enrollment...")
    # enroll users based on training data
    train_files, _ = load_and_split_data(DATASET_PATH, min_user_id=ENROLL_USERS_START, max_user_id=ENROLL_USERS_END)

    # create database
    enrollment_db = {}
    preprocessor = FingerprintPreprocessor()
    transform = SiameseFingerprintDataset.get_validation_transform()

    logging.info("Enrolling genuine users...")
    with torch.no_grad():
        for finger_id, file_paths in tqdm(train_files.items(), desc="Enrolling"):
            if not file_paths:
                continue

            normalized_embedding = generate_user_template(
                file_paths, embedding_net, preprocessor, transform, device, finger_id
            )
            if normalized_embedding is not None:
                enrollment_db[finger_id] = normalized_embedding

    # save database
    os.makedirs(os.path.dirname(ENROLLMENT_DB_PATH), exist_ok=True)
    with open(ENROLLMENT_DB_PATH, 'wb') as f:
        pickle.dump(enrollment_db, f)

    logging.info("Enrollment complete.")
    logging.info(f"Enrolled {len(enrollment_db)} unique fingers.")
    logging.info(f"Enrollment database saved to {ENROLLMENT_DB_PATH}")


if __name__ == '__main__':
    main()
