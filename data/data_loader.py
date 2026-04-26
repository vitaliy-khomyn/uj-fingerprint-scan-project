import os
import re
import logging
from collections import defaultdict
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# Module-level constant defining the subdirectories of the SOCOFing dataset to be scanned.
SOCOFING_SUBDIRS = [
    "Real",
    os.path.join("Altered", "Altered-Easy"),
    os.path.join("Altered", "Altered-Medium"),
    os.path.join("Altered", "Altered-Hard")
]


def parse_filename(filename):
    """
    Parses a SOCOFing filename to extract the person ID, gender, and finger name.

    Example: 1__M_Left_index_finger.BMP -> (1, 'M', 'Left_index_finger')
    Example: 100__F_Right_ring_finger_CR.BMP -> (100, 'F', 'Right_ring_finger')
    """
    # This regex is specifically designed to be robust against the SOCOFing naming scheme.
    # It explicitly matches the known finger names and separates them from alteration suffixes.
    finger_pattern = r"((?:Left|Right)_(?:thumb|index|middle|ring|little)(?:_finger)?)"
    match = re.match(rf"(\d+)__(\w)_{finger_pattern}(?:_.*)?\.BMP", filename, re.IGNORECASE)

    if match:
        person_id = int(match.group(1))
        gender = match.group(2)
        finger_name = match.group(3)
        return person_id, gender, finger_name
    return None, None, None


def _load_fingerprint_map(dataset_path, min_user_id, max_user_id):
    """
    Scans the dataset subdirectories and maps all found fingerprint image paths
    to their corresponding (person_id, finger_name) key.
    """
    fingerprint_map = defaultdict(list)
    logging.info(f"Scanning dataset root path: {dataset_path}")

    if not dataset_path or not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: '{dataset_path}'. Please check your .env file.")

    for sub_dir in SOCOFING_SUBDIRS:
        current_path = os.path.join(dataset_path, sub_dir)
        if not os.path.isdir(current_path):
            logging.warning(f"Subdirectory not found, skipping: {current_path}")
            continue

        logging.info(f"Scanning subdirectory: {current_path}")
        for filename in os.listdir(current_path):
            full_path = os.path.join(current_path, filename)
            person_id, _, finger_name = parse_filename(filename)

            if person_id is not None and min_user_id <= person_id <= max_user_id:
                fingerprint_map[(person_id, finger_name)].append(full_path)

    return fingerprint_map


def load_and_split_data(dataset_path, max_user_id=100, min_user_id=1, test_samples=5, random_state=42):
    """
    Loads fingerprint paths from the SOCOFing dataset and splits them into
    training and testing sets.

    Args:
        dataset_path (str): Absolute path to the root 'SOCOFing' directory.
        max_user_id (int): The maximum user ID to include.
        min_user_id (int): The minimum user ID to include.
        test_samples (int): The number of samples per finger to allocate to the test set.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (train_files, test_files)
               - train_files: A dictionary mapping (person_id, finger_name) to a list of file paths.
               - test_files: A dictionary mapping (person_id, finger_name) to a list of file paths.
    """
    fingerprint_map = _load_fingerprint_map(dataset_path, min_user_id, max_user_id)

    if not fingerprint_map:
        logging.warning(f"No fingerprint files were parsed in '{dataset_path}'. Please check the path and file naming.")
        return {}, {}

    logging.info(f"Found data for {len(fingerprint_map)} unique person/finger combinations from users {min_user_id}-{max_user_id}.")

    train_files = defaultdict(list)
    test_files = defaultdict(list)

    # Split the files for each finger into training and testing sets.
    for (person_id, finger_name), files in fingerprint_map.items():
        # Ensure there are enough samples for both training and testing.
        if len(files) > test_samples:
            train_set, test_set = train_test_split(files, test_size=test_samples, random_state=random_state)
            train_files[(person_id, finger_name)] = train_set
            test_files[(person_id, finger_name)] = test_set
        else:
            # If not enough samples, allocate all to training.
            train_files[(person_id, finger_name)] = files

    logging.info(f"Data split complete. Training samples: {sum(len(v) for v in train_files.values())}, Testing samples: {sum(len(v) for v in test_files.values())}")
    return dict(train_files), dict(test_files)


def main():
    """
    Runs a demonstration of the data loading and splitting process.
    This function is executed when the script is run directly.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
        if not DATASET_PATH:
            raise ValueError("SOCOFING_DATASET_PATH environment variable not set.")

        logging.info("Running data loader demonstration...")
        train_data, test_data = load_and_split_data(DATASET_PATH)

        if train_data:
            logging.info("--- Example Data ---")
            first_key = list(train_data.keys())[0]
            logging.info(f"Person/Finger: {first_key}")
            logging.info(f"  Training files: {len(train_data[first_key])}")
            logging.info(f"  Testing files: {len(test_data.get(first_key, []))}")
            logging.info(f"  Example file path: {train_data[first_key][0]}")

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"ERROR: {e}")
        logging.error("Please create a .env file (you can copy .env.sample) and set the 'SOCOFING_DATASET_PATH' variable to the correct location.")


if __name__ == '__main__':
    main()
