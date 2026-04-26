import os
import pickle
import logging
import torch
import numpy as np
from tqdm import tqdm
import cv2

from data.data_loader import load_and_split_data
from models.model import EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor
from data.dataset import SiameseFingerprintDataset

# Configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_PATH = "trained_models/embedding_net.pth"
ENROLLMENT_DB_PATH = "enrollment_db.pkl"
RESULTS_SAVE_PATH = "validation_results.pkl"

GENUINE_USERS_START = 401
GENUINE_USERS_END = 500
IMPOSTOR_USERS_START = 501
IMPOSTOR_USERS_END = 600


def calculate_metrics(genuine_distances, impostor_distances, threshold):
    """Calculates False Acceptance Rate (FAR) and False Rejection Rate (FRR) for a given threshold."""
    false_accepts = np.sum(np.array(impostor_distances) < threshold)
    false_rejects = np.sum(np.array(genuine_distances) > threshold)

    far = false_accepts / len(impostor_distances) if impostor_distances else 0
    frr = false_rejects / len(genuine_distances) if genuine_distances else 0
    return far, frr

def calculate_far_at_frr(genuine_distances, impostor_distances, target_frr):
    """Finds the FAR at a specific FRR target."""
    if not genuine_distances or not impostor_distances:
        return -1, -1
    
    # Find the threshold that yields the target FRR
    threshold = np.quantile(genuine_distances, 1 - target_frr)
    
    far, frr = calculate_metrics(genuine_distances, impostor_distances, threshold)
    return far, threshold

def get_embedding(image_path, model, preprocessor, transform, device):
    """Helper function to get a single image's embedding."""
    try:
        processed_img = preprocessor.preprocess(image_path)
        tensor_img = transform(processed_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor_img)
        return embedding.cpu().numpy()
    except Exception as e:
        logging.warning(f"Could not process file {image_path}. Error: {e}")
        return None


def main():
    cv2.setNumThreads(0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Setup and Loading
    # Provide more specific error messages for easier debugging.
    if not DATASET_PATH:
        raise FileNotFoundError("SOCOFING_DATASET_PATH not set in .env file. Please check your .env configuration.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please ensure train.py has run successfully.")
    if not os.path.exists(ENROLLMENT_DB_PATH):
        raise FileNotFoundError(f"Enrollment database not found at '{ENROLLMENT_DB_PATH}'. Please run enrollment.py after training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading model, enrollment database, and preprocessors...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)
    # Load the checkpoint dictionary and extract the model's state dict.
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    embedding_net.load_state_dict(checkpoint['model_state_dict'])
    embedding_net.eval()

    with open(ENROLLMENT_DB_PATH, 'rb') as f:
        enrollment_db = pickle.load(f)

    preprocessor = FingerprintPreprocessor(verbose=False)
    transform = SiameseFingerprintDataset.get_validation_transform()
    logging.info("Setup complete.")

    # 2. Collect Genuine and Impostor Distances
    logging.info("\n--- Collecting distances for analysis ---")

    # Collect Genuine Distances (1-to-1 verification)
    _, genuine_test_files = load_and_split_data(DATASET_PATH, min_user_id=GENUINE_USERS_START, max_user_id=GENUINE_USERS_END)
    genuine_distances = []
    for true_finger_id, file_paths in tqdm(genuine_test_files.items(), desc="Genuine Verification"):
        if true_finger_id not in enrollment_db:
            continue

        enrolled_embedding = enrollment_db[true_finger_id]
        for img_path in file_paths:
            embedding = get_embedding(img_path, embedding_net, preprocessor, transform, device)
            if embedding is None:
                continue

            # Calculate the distance between the test image and its correct enrolled template.
            distance = np.linalg.norm(embedding - enrolled_embedding)
            genuine_distances.append(distance)

    # Collect Impostor Distances
    impostor_train, impostor_test = load_and_split_data(DATASET_PATH, min_user_id=IMPOSTOR_USERS_START, max_user_id=IMPOSTOR_USERS_END)
    all_impostor_files = [file for files in list(impostor_train.values()) + list(impostor_test.values()) for file in files]
    impostor_distances = []
    for img_path in tqdm(all_impostor_files, desc="Impostor Verification"):
        embedding = get_embedding(img_path, embedding_net, preprocessor, transform, device)
        if embedding is None:
            continue

        # For an impostor, find the distance to the closest enrolled template in the entire database.
        min_distance = np.min([np.linalg.norm(embedding - enrolled_embedding) for enrolled_embedding in enrollment_db.values()])
        impostor_distances.append(min_distance)

    logging.info(f"Collected {len(genuine_distances)} genuine distances and {len(impostor_distances)} impostor distances.")

    # Save distances for later analysis and plotting.
    with open(RESULTS_SAVE_PATH, 'wb') as f:
        pickle.dump({'genuine': genuine_distances, 'impostor': impostor_distances}, f)
    logging.info(f"Validation distances saved to {RESULTS_SAVE_PATH}")

    # 3. Find Equal Error Rate (EER)
    logging.info("\n--- Calculating Equal Error Rate (EER) ---")
    min_diff = float('inf')
    eer = 0
    eer_threshold = 0

    # Iterate over a range of potential thresholds to find the EER point.
    thresholds = np.arange(0, 2.0, 0.001)
    for threshold in tqdm(thresholds, desc="Finding EER"):
        far, frr = calculate_metrics(genuine_distances, impostor_distances, threshold)

        if abs(far - frr) < min_diff:
            min_diff = abs(far - frr)
            # The EER is the point where FAR and FRR are closest.
            eer = (far + frr) / 2
            eer_threshold = threshold

    logging.info(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    logging.info(f"Optimal Threshold (at EER): {eer_threshold:.4f}")

    # 4. Report metrics at the optimal threshold
    far_at_eer, frr_at_eer = calculate_metrics(genuine_distances, impostor_distances, eer_threshold)
    logging.info("At this optimal threshold:")
    logging.info(f"  False Acceptance Rate (FAR): {far_at_eer * 100:.2f}%")
    logging.info(f"  False Rejection Rate (FRR): {frr_at_eer * 100:.2f}%")

    # 5. Report FAR at standard FRR benchmarks
    logging.info("\n--- System Performance at Standard FRR Benchmarks ---")
    benchmarks = {"FRR @ 10%": 0.10, "FRR @ 1%": 0.01, "FRR @ 0.1%": 0.001}
    for name, frr_target in benchmarks.items():
        far, threshold = calculate_far_at_frr(genuine_distances, impostor_distances, frr_target)
        if far != -1:
            logging.info(f"  {name}:")
            logging.info(f"    - FAR: {far * 100:.2f}%")
            logging.info(f"    - Threshold: {threshold:.4f}")


if __name__ == '__main__':
    main()
