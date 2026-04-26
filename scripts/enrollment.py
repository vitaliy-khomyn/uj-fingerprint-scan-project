import os
import pickle
import logging
import torch
from tqdm import tqdm
import numpy as np

from data.data_loader import load_and_split_data
from models.model import EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor
from data.dataset import SiameseFingerprintDataset

# Configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_PATH = "trained_models/embedding_net.pth"
ENROLLMENT_DB_PATH = "enrollment_db.pkl"
ENROLL_USERS_START = 401
ENROLL_USERS_END = 500


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup
    if not DATASET_PATH:
        raise ValueError("SOCOFING_DATASET_PATH environment variable not set. Please create a .env file.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    logging.info("Loading trained embedding model...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)
    # Load the checkpoint dictionary and extract the model's state dict.
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    embedding_net.load_state_dict(checkpoint['model_state_dict'])
    embedding_net.eval()
    logging.info("Model loaded successfully.")

    # Load data
    logging.info("Loading data for enrollment...")
    # We will enroll users based on their training data samples for robustness.
    train_files, _ = load_and_split_data(DATASET_PATH, min_user_id=ENROLL_USERS_START, max_user_id=ENROLL_USERS_END)

    # Create enrollment database
    enrollment_db = {}
    # Use a non-verbose preprocessor for cleaner logs
    preprocessor = FingerprintPreprocessor(verbose=False)
    # Get the standard validation transforms, as enrollment is an inference task.
    transform = SiameseFingerprintDataset.get_validation_transform()

    logging.info("Enrolling genuine users...")
    with torch.no_grad():
        for finger_id, file_paths in tqdm(train_files.items(), desc="Enrolling"):
            if not file_paths:
                continue

            embeddings = []
            for img_path in file_paths:
                try:
                    # Preprocess and transform the image
                    processed_img = preprocessor.preprocess(img_path)
                    tensor_img = transform(processed_img).unsqueeze(0).to(device)

                    embedding = embedding_net(tensor_img)
                    embeddings.append(embedding.cpu().numpy())
                except Exception as e:
                    logging.warning(f"Could not process file {img_path}. Error: {e}")

            if not embeddings:
                logging.warning(f"No valid embeddings generated for {finger_id}. Skipping.")
                continue

            # Calculate the mean embedding for the finger to create a robust template
            mean_embedding = np.mean(embeddings, axis=0)

            # L2-normalize the final template vector
            norm = np.linalg.norm(mean_embedding)
            normalized_embedding = mean_embedding / norm if norm != 0 else mean_embedding

            enrollment_db[finger_id] = normalized_embedding

    # Save database
    with open(ENROLLMENT_DB_PATH, 'wb') as f:
        pickle.dump(enrollment_db, f)

    logging.info("Enrollment complete.")
    logging.info(f"Enrolled {len(enrollment_db)} unique fingers.")
    logging.info(f"Enrollment database saved to {ENROLLMENT_DB_PATH}")


if __name__ == '__main__':
    main()
