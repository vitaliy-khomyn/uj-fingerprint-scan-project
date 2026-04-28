import cv2
import logging
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from typing import Any, List, Optional, Tuple

from data.data_loader import load_and_split_data
from data.dataset import SiameseFingerprintDataset
from models.model import EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor

# configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("results", "trained_models", "embedding_net.pth"))
ENROLLMENT_DB_PATH = os.getenv("ENROLLMENT_DB_PATH", os.path.join("results", "enrollments", "enrollment_db.pkl"))
RESULTS_SAVE_PATH = os.getenv("RESULTS_SAVE_PATH", os.path.join("results", "validations", "validation_results.pkl"))

GENUINE_USERS_START = 401
GENUINE_USERS_END = 500
IMPOSTOR_USERS_START = 501
IMPOSTOR_USERS_END = 600


def calculate_metrics(genuine_distances: List[float], impostor_distances: List[float], threshold: float) -> Tuple[float, float]:
    """
    Calculates the False Acceptance Rate and False Rejection Rate for a given threshold.

    Args:
        genuine_distances (List[float]): Distances between genuine pairs evaluated.
        impostor_distances (List[float]): Distances between impostor pairs evaluated.
        threshold (float): The target threshold utilized for metric bounds calculation.

    Returns:
        Tuple[float, float]: Generated metrics tuple containing (far, frr).
    """
    false_accepts = np.sum(np.array(impostor_distances) < threshold)
    false_rejects = np.sum(np.array(genuine_distances) > threshold)

    far = false_accepts / len(impostor_distances) if impostor_distances else 0
    frr = false_rejects / len(genuine_distances) if genuine_distances else 0
    return far, frr


def calculate_far_at_frr(genuine_distances: List[float], impostor_distances: List[float], target_frr: float) -> Tuple[float, float]:
    """
    Finds the False Acceptance Rate calculated against a targeted specific FRR.

    Args:
        genuine_distances (List[float]): Ground genuine predictions collected.
        impostor_distances (List[float]): Ground impostor predictions collected.
        target_frr (float): The benchmark FRR limit.

    Returns:
        Tuple[float, float]: Calculated constraint pair indicating (far, specific_threshold).
    """
    if not genuine_distances or not impostor_distances:
        return -1, -1

    # find threshold that yields target frr
    threshold = np.quantile(genuine_distances, 1 - target_frr)

    far, frr = calculate_metrics(genuine_distances, impostor_distances, threshold)
    return far, threshold


def get_embedding(image_path: str, model: torch.nn.Module, preprocessor: FingerprintPreprocessor, transform: Any, device: torch.device) -> Optional[np.ndarray]:
    """
    Generates the embedding array equivalent for a single provided image payload.

    Args:
        image_path (str): Reference string pointing to absolute system path.
        model (torch.nn.Module): Executed CNN evaluated logic layer.
        preprocessor (FingerprintPreprocessor): Pipeline transformer class instance.
        transform (Any): Sequential application functions provided via composition.
        device (torch.device): Compute boundary node initialized representation.

    Returns:
        Optional[np.ndarray]: Respective 1D array normalized representation of image features.
    """
    try:
        processed_img = preprocessor.preprocess(image_path)
        tensor_img = transform(processed_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor_img)
        return embedding.cpu().numpy()
    except Exception as e:
        logging.warning(f"Could not process file {image_path}. Error: {e}")
        return None


def collect_genuine_distances(dataset_path: str, enrollment_db: dict, model: torch.nn.Module, preprocessor: FingerprintPreprocessor, transform: Any, device: torch.device) -> List[float]:
    """
    Calculates distances between tests and genuine enrolled user templates evaluated systematically.

    Args:
        dataset_path (str): Parent origin target dataset location context.
        enrollment_db (dict): Saved dictionary pairing user id key mappings to embeddings vectors.
        model (torch.nn.Module): Current generated network model processing input arrays.
        preprocessor (FingerprintPreprocessor): Logic pipeline module responsible for image corrections.
        transform (Any): Normalization tensor wrapper format configurations passed functionally.
        device (torch.device): Processor bound reference utilized dynamically.

    Returns:
        List[float]: The calculated float magnitudes mapped globally against genuine metrics.
    """
    _, genuine_test_files = load_and_split_data(dataset_path, min_user_id=GENUINE_USERS_START, max_user_id=GENUINE_USERS_END)
    genuine_distances = []
    for true_finger_id, file_paths in tqdm(genuine_test_files.items(), desc="Genuine Verification"):
        if true_finger_id not in enrollment_db:
            continue

        enrolled_embedding = enrollment_db[true_finger_id]
        for img_path in file_paths:
            embedding = get_embedding(img_path, model, preprocessor, transform, device)
            if embedding is None:
                continue

            # calculate distance between test image and enrolled template
            distance = np.linalg.norm(embedding - enrolled_embedding)
            genuine_distances.append(distance)
    return genuine_distances


def collect_impostor_distances(dataset_path: str, enrollment_db: dict, model: torch.nn.Module, preprocessor: FingerprintPreprocessor, transform: Any, device: torch.device) -> List[float]:
    """
    Calculates distances mapping target distances spanning test sets against generic optimal templates uniformly.

    Args:
        dataset_path (str): Reference string pointing to overall test configurations globally applied.
        enrollment_db (dict): Authenticated templates registered locally during previous generation.
        model (torch.nn.Module): Convolution representation of network evaluation.
        preprocessor (FingerprintPreprocessor): Pipeline normalization sequence algorithm layer.
        transform (Any): Target conversion metric transformations applied procedurally.
        device (torch.device): Device runtime context.

    Returns:
        List[float]: Array grouping representation mapping values calculated as impostor distances.
    """
    impostor_train, impostor_test = load_and_split_data(dataset_path, min_user_id=IMPOSTOR_USERS_START, max_user_id=IMPOSTOR_USERS_END)
    all_impostor_files = [file for files in list(impostor_train.values()) + list(impostor_test.values()) for file in files]
    impostor_distances = []
    for img_path in tqdm(all_impostor_files, desc="Impostor Verification"):
        embedding = get_embedding(img_path, model, preprocessor, transform, device)
        if embedding is None:
            continue

        # find distance to closest enrolled template in database
        min_distance = np.min([np.linalg.norm(embedding - enrolled_embedding) for enrolled_embedding in enrollment_db.values()])
        impostor_distances.append(min_distance)
    return impostor_distances


def find_eer(genuine_distances: List[float], impostor_distances: List[float]) -> Tuple[float, float]:
    """
    Finds the exact evaluation target equal error rate intercept mapped locally alongside valid thresholds.

    Args:
        genuine_distances (List[float]): Calculated set containing correct ground truth distance representations.
        impostor_distances (List[float]): Incorrect baseline boundaries formulated explicitly iteratively.

    Returns:
        Tuple[float, float]: Representation format corresponding to matching evaluation values `(eer, threshold)`.
    """
    min_diff = float('inf')
    eer = 0
    eer_threshold = 0

    # iterate over potential thresholds to find eer point
    thresholds = np.arange(0, 2.0, 0.001)
    for threshold in tqdm(thresholds, desc="Finding EER"):
        far, frr = calculate_metrics(genuine_distances, impostor_distances, threshold)

        if abs(far - frr) < min_diff:
            min_diff = abs(far - frr)
            eer = (far + frr) / 2
            eer_threshold = threshold
    return eer, eer_threshold


def main() -> None:
    """
    Executes primary metric assessment logic sequentially running entire environment sequentially.

    Returns:
        None
    """
    cv2.setNumThreads(0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # setup and loading
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
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    embedding_net.load_state_dict(checkpoint['model_state_dict'])
    embedding_net.eval()

    with open(ENROLLMENT_DB_PATH, 'rb') as f:
        enrollment_db = pickle.load(f)

    preprocessor = FingerprintPreprocessor()
    transform = SiameseFingerprintDataset.get_validation_transform()
    logging.info("Setup complete.")

    # collect genuine and impostor distances
    logging.info("\n--- Collecting distances for analysis ---")

    genuine_distances = collect_genuine_distances(DATASET_PATH, enrollment_db, embedding_net, preprocessor, transform, device)

    impostor_distances = collect_impostor_distances(DATASET_PATH, enrollment_db, embedding_net, preprocessor, transform, device)

    logging.info(f"Collected {len(genuine_distances)} genuine distances and {len(impostor_distances)} impostor distances.")

    # save distances
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)
    with open(RESULTS_SAVE_PATH, 'wb') as f:
        pickle.dump({'genuine': genuine_distances, 'impostor': impostor_distances}, f)
    logging.info(f"Validation distances saved to {RESULTS_SAVE_PATH}")

    # find equal error rate
    logging.info("\n--- Calculating Equal Error Rate (EER) ---")
    eer, eer_threshold = find_eer(genuine_distances, impostor_distances)

    logging.info(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    logging.info(f"Optimal Threshold (at EER): {eer_threshold:.4f}")

    # report metrics
    far_at_eer, frr_at_eer = calculate_metrics(genuine_distances, impostor_distances, eer_threshold)
    logging.info("At this optimal threshold:")
    logging.info(f"  False Acceptance Rate (FAR): {far_at_eer * 100:.2f}%")
    logging.info(f"  False Rejection Rate (FRR): {frr_at_eer * 100:.2f}%")

    # report far at benchmarks
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
